"""
RetinaScreen - Diabetic Retinopathy stage detection demo (Streamlit).

Run locally:      streamlit run app/streamlit_app.py
Requires:         models/best_model.keras and models/model_metadata.json
                  (exported by the Kaggle notebook, or by scripts/train.py --save-best)

Tabs
----
1. Screen an image  - upload a fundus photo -> preprocessing preview, predicted
                      grade + probabilities, Grad-CAM heat-map, referral advice
2. Batch screening  - grade many images at once and download a CSV report
3. RetinaBot        - rule-based assistant that explains results and DR stages
4. Model card       - dataset, architecture, metrics and limitations
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# make `dr_detection` importable when the app is started from the repo root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

from dr_detection import config as C  # noqa: E402
from dr_detection.gradcam import make_gradcam_heatmap, overlay_heatmap  # noqa: E402
from dr_detection.preprocessing import preprocess_image, preprocessing_stages  # noqa: E402
from dr_detection import kaggle_pipeline as KP  # noqa: E402
from dr_detection.utils import load_json  # noqa: E402
from chatbot import STAGE_INFO, ChatContext, RetinaBot  # noqa: E402

MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "figures"
SAMPLES_DIR = ROOT / "app" / "assets" / "samples"
GRADE_COLOURS = ["#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#9b2226"]

st.set_page_config(page_title="RetinaScreen - DR stage detection", page_icon="👁️", layout="wide")


# --------------------------------------------------------------------------- #
# Model loading (cached across reruns)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading the trained model ...")
def load_model_and_meta():
    from tensorflow import keras

    meta_path = MODELS_DIR / "model_metadata.json"
    model_path = MODELS_DIR / "best_model.keras"
    if not model_path.exists() or not meta_path.exists():
        return None, None
    model = keras.models.load_model(model_path)
    meta = load_json(meta_path)
    # warm-up: the first call traces the graph and is slow; do it once here so the
    # first real prediction in a demo is fast
    dummy = np.zeros((meta["img_size"], meta["img_size"], 3), dtype=np.uint8)
    make_gradcam_heatmap(model, dummy)
    return model, meta


def pil_to_rgb(file) -> np.ndarray:
    return np.array(Image.open(file).convert("RGB"))


def uses_kaggle_pipeline(meta) -> bool:
    """Models trained in the Kaggle notebook store their preprocessing specification."""
    return bool(meta) and "preprocessing_spec" in meta


def preprocess_for_model(rgb: np.ndarray, meta) -> np.ndarray:
    """Apply exactly the preprocessing the deployed model was trained with."""
    if uses_kaggle_pipeline(meta):
        return KP.preprocess(rgb, meta["preprocess"], meta["img_size"])
    return preprocess_image(rgb, meta["preprocess"], meta["img_size"])


def pipeline_stages(rgb: np.ndarray, meta):
    if uses_kaggle_pipeline(meta):
        return KP.stages(rgb, meta["img_size"])
    return preprocessing_stages(rgb, size=meta["img_size"])


def metric(m: dict, key: str) -> float:
    """Metric lookup that accepts both the notebook and the older script key names."""
    aliases = {"qwk": ["qwk", "quadratic_weighted_kappa"]}
    for k in aliases.get(key, [key]):
        if k in m:
            return float(m[k])
    return float("nan")


def grade_image(model, meta, rgb: np.ndarray):
    """Preprocess -> predict (with test-time augmentation if the model was validated with it)
    -> Grad-CAM.  Returns a dict used by the UI."""
    t0 = time.perf_counter()
    proc = preprocess_for_model(rgb, meta)
    heatmap, probs, k = make_gradcam_heatmap(model, proc)
    if meta.get("tta"):
        views = np.stack([proc, proc[:, ::-1], proc[::-1], proc[::-1, ::-1]]).astype(np.float32)
        probs = model.predict(views, verbose=0).mean(axis=0)
        k = int(np.argmax(probs))
    return {
        "processed": proc,
        "probs": probs,
        "grade": int(k),
        "confidence": float(probs[k]),
        "heatmap": heatmap,
        "overlay": overlay_heatmap(proc, heatmap),
        "ms": (time.perf_counter() - t0) * 1000,
    }


def prob_chart(probs: np.ndarray):
    """Horizontal bar chart of the five grade probabilities, in grade order."""
    import altair as alt

    df = pd.DataFrame({"grade": [f"{i} - {C.CLASS_LABELS[i]}" for i in range(5)],
                       "probability": probs, "colour": GRADE_COLOURS})
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            y=alt.Y("grade:N", sort=None, title=None),
            color=alt.Color("colour:N", scale=None, legend=None),
            tooltip=[alt.Tooltip("grade:N"), alt.Tooltip("probability:Q", format=".1%")],
        )
        .properties(height=200)
    )
    st.altair_chart(chart, width="stretch")


def screening_summary(name: str, res: dict, meta: dict) -> str:
    """Plain-text summary of one screening, for the download button."""
    s = STAGE_INFO[res["grade"]]
    lines = [
        "RetinaScreen - diabetic retinopathy screening summary",
        f"Image: {name}",
        f"Date: {time.strftime('%Y-%m-%d %H:%M')}",
        f"Model: {meta['backbone']} ({meta['preprocess']} preprocessing, {meta['img_size']} px)",
        "",
        f"Predicted grade: {res['grade']} - {s['name']}",
        f"Confidence: {res['confidence']:.1%}",
        f"Referable DR (grade >= 2): {'yes' if res['grade'] >= C.REFERABLE_THRESHOLD else 'no'}",
        f"Urgency: {s['urgency']}",
        f"Recommended action: {s['advice']}",
        "",
        "Grade probabilities:",
    ] + [f"  {i} - {C.CLASS_LABELS[i]:18s} {p:.1%}" for i, p in enumerate(res["probs"])] + [
        "",
        "This is a coursework prototype, not a medical device. Results must be confirmed by an eye-care professional.",
    ]
    return "\n".join(lines)


def referral_banner(grade: int, confidence: float) -> None:
    s = STAGE_INFO[grade]
    text = f"**Grade {grade} - {s['name']}** ({confidence:.0%} confidence). {s['advice'].capitalize()}"
    if grade == 0:
        st.success(text)
    elif grade == 1:
        st.info(text)
    elif grade == 2:
        st.warning(text)
    else:
        st.error(text)
    if confidence < 0.55:
        st.caption("Low confidence: the image lies between two grades and should be reviewed by a human grader.")


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
model, meta = load_model_and_meta()

with st.sidebar:
    st.title("👁️ RetinaScreen")
    st.caption("Diabetic Retinopathy stage detection with transfer learning")
    if meta:
        st.markdown(f"**Model:** {meta['backbone']}  \n**Preprocessing:** {meta['preprocess']}  \n"
                    f"**Input size:** {meta['img_size']} px")
        m = meta.get("metrics", {})
        if m:
            st.markdown("**Internal test (APTOS 2019)**")
            c1, c2 = st.columns(2)
            c1.metric("Accuracy", f"{metric(m, 'accuracy'):.1%}")
            c2.metric("QWK", f"{metric(m, 'qwk'):.3f}")
            c1.metric("Macro F1", f"{metric(m, 'macro_f1'):.3f}")
            c2.metric("Referable sens.", f"{metric(m, 'referable_sensitivity'):.1%}")
        ext = meta.get("external_metrics", {})
        if ext:
            st.markdown("**External validation (EyePACS 2015)**")
            c1, c2 = st.columns(2)
            c1.metric("Accuracy", f"{metric(ext, 'accuracy'):.1%}")
            c2.metric("QWK", f"{metric(ext, 'qwk'):.3f}")
            c1.metric("Macro F1", f"{metric(ext, 'macro_f1'):.3f}")
            c2.metric("Referable sens.", f"{metric(ext, 'referable_sensitivity'):.1%}")
    else:
        st.error("No trained model found in `models/`. Run `scripts/train.py --save-best` first.")
    st.divider()
    st.markdown("**Grades (ICDR scale)**")
    for g, s in STAGE_INFO.items():
        st.markdown(f"<span style='color:{GRADE_COLOURS[g]}'>&#9632;</span> {g} - {s['name']}", unsafe_allow_html=True)
    st.divider()
    st.caption("Coursework prototype - not a medical device.")

tab_single, tab_batch, tab_bot, tab_card = st.tabs(["🔍 Screen an image", "📂 Batch screening", "💬 RetinaBot", "📋 Model card"])

# ---------------------------------------------------------------------- #
# Tab 1 - single image
# ---------------------------------------------------------------------- #
with tab_single:
    st.subheader("Screen a fundus photograph")
    col_up, col_opts = st.columns([2, 1])
    with col_up:
        file = st.file_uploader("Upload a colour fundus image (PNG / JPG)", type=["png", "jpg", "jpeg"])
        samples = sorted(SAMPLES_DIR.glob("*.png")) + sorted(SAMPLES_DIR.glob("*.jpg")) if SAMPLES_DIR.exists() else []
        sample_choice = None
        if samples:
            sample_choice = st.selectbox("... or pick a bundled sample image",
                                         ["(none)"] + [p.name for p in samples], index=0)
    with col_opts:
        show_stages = st.checkbox("Show preprocessing stages", value=True)
        alpha = st.slider("Heat-map opacity", 0.2, 0.7, 0.4, 0.05)

    if file is None and sample_choice and sample_choice != "(none)":
        file = SAMPLES_DIR / sample_choice

    if file is not None and model is not None:
        rgb = pil_to_rgb(file)
        with st.spinner("Grading ..."):
            res = grade_image(model, meta, rgb)
        res["overlay"] = overlay_heatmap(res["processed"], res["heatmap"], alpha)
        st.session_state["last_result"] = res

        referral_banner(res["grade"], res["confidence"])

        c1, c2, c3 = st.columns(3)
        c1.image(rgb, caption=f"Input image ({rgb.shape[1]}x{rgb.shape[0]})", width="stretch")
        c2.image(res["processed"], caption=f"Preprocessed ({meta['preprocess']}, {meta['img_size']} px)", width="stretch")
        c3.image(res["overlay"], caption="Grad-CAM: regions that drove the prediction", width="stretch")

        c4, c5 = st.columns([1, 1])
        with c4:
            st.markdown("**Probability of each grade**")
            prob_chart(res["probs"])
        with c5:
            st.markdown("**Details**")
            st.write(pd.DataFrame({
                "grade": [f"{i} - {C.CLASS_LABELS[i]}" for i in range(5)],
                "probability": [f"{p:.1%}" for p in res["probs"]],
            }).set_index("grade"))
            referable = res["grade"] >= C.REFERABLE_THRESHOLD
            st.markdown(f"**Referable DR:** {'Yes' if referable else 'No'}  \n"
                        f"**Inference time:** {res['ms']:.0f} ms")
            summary = screening_summary(getattr(file, "name", str(file)), res, meta)
            st.download_button("Download screening summary", summary, "screening_summary.txt", "text/plain")

        if show_stages:
            st.markdown("**Preprocessing pipeline applied to this image**")
            stages = pipeline_stages(rgb, meta)
            cols = st.columns(len(stages))
            for col, (name, img) in zip(cols, stages.items()):
                col.image(img, caption=name, width="stretch", clamp=True)
    elif file is not None and model is None:
        st.error("Model not available.")
    else:
        st.info("Upload an image to begin. Sample images are available in the `app/assets/samples/` folder of the repository.")

# ---------------------------------------------------------------------- #
# Tab 2 - batch
# ---------------------------------------------------------------------- #
with tab_batch:
    st.subheader("Batch screening")
    files = st.file_uploader("Upload several fundus images", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="batch")
    if files and model is not None:
        rows, thumbs = [], []
        progress = st.progress(0.0)
        for i, f in enumerate(files):
            res = grade_image(model, meta, pil_to_rgb(f))
            rows.append({
                "image": f.name, "grade": res["grade"], "class": C.CLASS_LABELS[res["grade"]],
                "confidence": round(res["confidence"], 3),
                "referable": res["grade"] >= C.REFERABLE_THRESHOLD,
                "urgency": STAGE_INFO[res["grade"]]["urgency"],
                **{f"p_{C.CLASS_NAMES[k]}": round(float(res["probs"][k]), 3) for k in range(5)},
            })
            thumbs.append((f.name, res["overlay"], res["grade"], res["confidence"]))
            progress.progress((i + 1) / len(files))
        df = pd.DataFrame(rows)
        st.dataframe(df, width="stretch")
        n_ref = int(df["referable"].sum())
        st.markdown(f"**{len(df)} images graded - {n_ref} referable ({n_ref/len(df):.0%}).**")
        st.download_button("Download CSV report", df.to_csv(index=False).encode(), "dr_batch_report.csv", "text/csv")
        st.markdown("**Grad-CAM overlays**")
        cols = st.columns(min(6, len(thumbs)))
        for i, (name, img, g, conf) in enumerate(thumbs):
            cols[i % len(cols)].image(img, caption=f"{name}: grade {g} ({conf:.0%})", width="stretch")

# ---------------------------------------------------------------------- #
# Tab 3 - chatbot
# ---------------------------------------------------------------------- #
with tab_bot:
    st.subheader("RetinaBot - ask about your result or about diabetic retinopathy")
    bot = RetinaBot()
    if "chat" not in st.session_state:
        st.session_state["chat"] = [("assistant", bot.greeting(ChatContext()))]
    last = st.session_state.get("last_result")
    ctx = ChatContext(
        grade=last["grade"] if last else None,
        confidence=last["confidence"] if last else None,
        probabilities=[float(p) for p in last["probs"]] if last else None,
        model_name=meta["backbone"] if meta else "CNN",
        metrics=meta.get("metrics", {}) if meta else {},
        external_metrics=meta.get("external_metrics", {}) if meta else {},
    )
    for role, text in st.session_state["chat"]:
        with st.chat_message(role):
            st.markdown(text)
    quick = st.columns(4)
    prompts = ["What does my result mean?", "What should I do next?", "What are the DR stages?", "What is the heat-map?"]
    clicked = None
    for col, p in zip(quick, prompts):
        if col.button(p, width="stretch"):
            clicked = p
    user_msg = st.chat_input("Type a question ...") or clicked
    if user_msg:
        st.session_state["chat"].append(("user", user_msg))
        st.session_state["chat"].append(("assistant", bot.reply(user_msg, ctx)))
        st.rerun()

# ---------------------------------------------------------------------- #
# Tab 4 - model card
# ---------------------------------------------------------------------- #
with tab_card:
    st.subheader("Model card")
    if meta:
        tc = meta.get("training_config", {})
        ds = meta.get("datasets", {})
        st.markdown(f"""
**Task** - 5-class diabetic retinopathy grading (ICDR scale) from colour fundus photographs.

**Development data** - {ds.get("development", "APTOS 2019 Blindness Detection (Kaggle)")}: duplicate
photographs removed, stratified 70 / 15 / 15 train / validation / test split, seed 42.

**External validation data** - {ds.get("external", "not evaluated")}: never used for training or tuning.

**Architecture** - {meta['backbone']} (ImageNet weights, `include_top=False`) -> GlobalAveragePooling ->
Dropout {tc.get("dropout", "")} -> Dense 512 -> Dropout -> Dense 256 -> Dense 5 (softmax).

**Preprocessing** - black-border crop -> pad to square -> resize {meta['img_size']} px ->
`{meta['preprocess']}` enhancement -> retina mask. Test-time augmentation (flips): {"yes" if meta.get("tta") else "no"}.

**Training** - phase 1: frozen backbone, Adam {tc.get("lr_head", "")}; phase 2: fine-tuning
({tc.get("unfreeze", "")} layers, BatchNorm frozen), Adam {tc.get("lr_ft", "")}; batch size {tc.get("batch", "")};
class balancing: {tc.get("balance", "")}; on-the-fly augmentation (flips, rotation, zoom, shift,
brightness, contrast); early stopping on validation QWK, ReduceLROnPlateau, model checkpointing.
""")
        m, ext = meta.get("metrics", {}), meta.get("external_metrics", {})
        if m:
            table = pd.DataFrame({"APTOS 2019 test (internal)": pd.Series(m)})
            if ext:
                table["EyePACS 2015 (external)"] = pd.Series(ext)
            st.markdown("**Performance**")
            st.table(table.apply(lambda col: col.map(lambda v: f"{v:.4f}")))
        hw = meta.get("trained_on", {})
        if hw:
            st.caption(f"Trained with TensorFlow {hw.get('tensorflow')} on {hw.get('gpu_name') or hw.get('gpu') or 'CPU'}.")
    # Training and evaluation evidence exported by the Kaggle notebook (figures/ folder).
    evidence = [
        (FIGURES_DIR / "09_experiment_results.png", "Experiments on the validation subset (one factor per stage)"),
        (FIGURES_DIR / "09_curves_final.png", "Accuracy and loss curves of the final model"),
        (FIGURES_DIR / "10_confusion_matrices.png", "Confusion matrices on the internal test subset"),
        (FIGURES_DIR / "10_per_class_metrics_and_roc.png", "Per-grade precision, recall, F1 and ROC curves"),
        (FIGURES_DIR / "10_confidence_calibration.png", "Confidence calibration"),
        (FIGURES_DIR / "10_gradcam_test.png", "Grad-CAM on test images"),
        (FIGURES_DIR / "11_internal_vs_external.png", "Internal test vs external validation"),
        (FIGURES_DIR / "11_external_confusion_matrices.png", "External validation confusion matrices"),
    ]
    evidence = [(p, cap) for p, cap in evidence if p.exists()]
    if evidence:
        st.markdown("**Training and evaluation evidence**")
        for i in range(0, len(evidence), 2):
            cols = st.columns(2)
            for col, (p, cap) in zip(cols, evidence[i:i + 2]):
                col.image(str(p), caption=cap, width="stretch")
    st.markdown("""
**Intended use** - decision *support* for screening programmes: prioritise which patients an
ophthalmologist should see first.  Not a diagnostic device.

**Limitations** - developed on one source (Aravind Eye Hospital, India) and validated externally
on EyePACS (USA); cannot detect diabetic macular oedema or non-DR pathology; adjacent grades are
frequently confused; image quality strongly affects results; model confidence is not clinical certainty.

**Ethics** - both datasets are anonymised and released under CC0; predictions are explained with
Grad-CAM so a clinician can verify that the evidence is anatomically plausible.
""")

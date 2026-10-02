"""
RetinaScreen - Diabetic Retinopathy stage detection demo (Streamlit).

Run locally:      streamlit run app/streamlit_app.py
Requires:         models/best_model.keras and models/model_metadata.json
                  (exported by the Kaggle notebook, or by scripts/train.py --save-best)

Tabs
----
1. Screen a photo   - upload a fundus photo or pick a real test photo -> stage on the
                      ICDR scale, urgency and what to do next. Collapsed sections show
                      the Grad-CAM heat-map, the stage probabilities and technical details
2. Batch screening  - grade many images at once, sorted by urgency, with a CSV report
3. Ask RetinaBot    - rule-based assistant that explains results and DR stages
4. About the model  - accuracy in plain language; model card, metrics and evidence
                      figures in collapsed sections
"""

from __future__ import annotations

import io
import re
import sys
import time
from html import escape
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
GRADE_COLOURS = ["#1E9E8A", "#D9A21B", "#E8772E", "#D1453B", "#8E1B24"]
LOW_CONFIDENCE = 0.55          # below this the image is flagged for a human grader
SAMPLE_RE = re.compile(r"^(aptos|eyepacs)_grade(\d)_", re.IGNORECASE)

st.set_page_config(page_title="RetinaScreen - DR stage detection", page_icon="👁️", layout="wide",
                   initial_sidebar_state="collapsed")


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


@st.cache_data(show_spinner=False, max_entries=64)
def grade_bytes(data: bytes, run_name: str, _model, _meta) -> dict:
    """Grade one encoded image. Cached on the image bytes, so reruns caused by the chat,
    the opacity slider or tab changes do not grade the same photograph again."""
    rgb = np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    res = grade_image(_model, _meta, rgb)
    res["rgb"] = rgb
    return res


@st.cache_data(show_spinner=False)
def thumbnail(path: str, grade=None, size: int = 160) -> np.ndarray:
    """Small preview of a sample; its reference grade is drawn as a coloured badge in the corner."""
    from PIL import ImageDraw, ImageFont

    img = Image.open(path).convert("RGB")
    img.thumbnail((size, size))
    if grade is not None:
        draw, r = ImageDraw.Draw(img), size // 7
        draw.ellipse((4, 4, 4 + 2 * r, 4 + 2 * r), fill=GRADE_COLOURS[grade], outline="white", width=2)
        draw.text((4 + r, 4 + r), str(grade), fill="white", anchor="mm", font=ImageFont.load_default(size=int(1.3 * r)))
    return np.array(img)


def sample_info(name: str):
    """Source dataset and reference grade encoded in a bundled sample's file name."""
    m = SAMPLE_RE.match(name)
    if not m:
        return None, None
    source = "APTOS 2019 test set" if m.group(1).lower() == "aptos" else "EyePACS 2015 external set"
    return source, int(m.group(2))


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


# --------------------------------------------------------------------------- #
# Presentation helpers (HTML is generated here; user-supplied text is escaped)
# --------------------------------------------------------------------------- #
CSS = """
<style>
[data-testid="stMainBlockContainer"], .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1200px;}
[data-testid="stHeader"] {background: transparent;}
.stTabs [data-baseweb="tab-list"] {gap: 4px;}
.stTabs [data-baseweb="tab"] {height: 46px; padding: 0 16px;}
.stTabs [data-baseweb="tab"] p {font-size: 0.98rem; font-weight: 600;}
[data-testid="stExpander"] summary p {font-weight: 600;}

.rs-hero {display:flex; align-items:center; gap:16px; padding:14px 22px; margin:0 0 4px; border-radius:16px; color:#fff;
  background:linear-gradient(120deg,#0F766E 0%,#115E59 58%,#164E63 100%); box-shadow:0 8px 24px rgba(15,118,110,.16);}
.rs-logo {font-size:2rem; line-height:1; background:rgba(255,255,255,.14); border-radius:12px; padding:8px 10px;}
.rs-title {font-size:1.55rem; font-weight:750; letter-spacing:-.02em; line-height:1.15;}
.rs-sub {color:#CCFBF1; margin-top:2px; font-size:.98rem;}

.rs-step-h {display:flex; align-items:center; gap:10px; font-weight:700; font-size:1.05rem; color:#0F172A; margin:2px 0 8px;}
.rs-step-h span {display:inline-flex; align-items:center; justify-content:center; width:26px; height:26px; border-radius:999px;
  background:#0F766E; color:#fff; font-size:.82rem; flex:none;}
.rs-muted {color:#64748B; font-size:.88rem;}

.rs-how {display:grid; gap:10px;}
.rs-how div {display:flex; gap:12px; align-items:flex-start; background:#fff; border:1px solid #E2E8F0; border-radius:12px; padding:12px 14px;}
.rs-how i {font-style:normal; font-size:1.4rem; line-height:1.2;}
.rs-how b {display:block; color:#0F172A;}
.rs-how span {color:#64748B; font-size:.86rem;}
.rs-how p {margin:0;}

.rs-result {background:#fff; border:1px solid #E2E8F0; border-left:7px solid var(--g); border-radius:14px;
  padding:18px 22px 16px; margin-bottom:12px; box-shadow:0 1px 3px rgba(15,23,42,.05);}
.rs-kicker {font-size:.72rem; font-weight:700; letter-spacing:.09em; text-transform:uppercase; color:#64748B;}
.rs-grade {font-size:1.6rem; font-weight:750; color:#0F172A; letter-spacing:-.01em; margin:2px 0 10px; line-height:1.25;}
.rs-grade em {font-style:normal; color:var(--g);}
.rs-chip {display:inline-block; padding:4px 12px; margin:0 6px 6px 0; border-radius:999px; font-size:.84rem; font-weight:650;}
.rs-advice {color:#1E293B; margin:8px 0 2px; font-size:1rem; background:#F8FAFC; border-radius:10px; padding:10px 12px;}
.rs-warn {background:#FFF7ED; border:1px solid #FED7AA; color:#9A3412; border-radius:10px; padding:8px 12px; font-size:.88rem; margin-top:10px;}
.rs-scale {display:grid; grid-template-columns:repeat(5,1fr); gap:6px; margin-top:16px; align-items:end;}
.rs-seg .bar {height:8px; border-radius:6px; opacity:.25;}
.rs-seg.on .bar {height:14px; opacity:1;}
.rs-seg .lbl {font-size:.72rem; color:#94A3B8; text-align:center; margin-top:5px; line-height:1.2;}
.rs-seg.on .lbl {color:#0F172A; font-weight:700;}

.rs-photo-bar {display:flex; align-items:center; gap:8px; color:#334155; font-size:.92rem; padding:4px 0;}

.rs-prob {display:grid; grid-template-columns:150px 1fr 56px; align-items:center; gap:12px; margin:8px 0; font-size:.9rem; color:#475569;}
.rs-prob b {text-align:right; font-weight:600; color:#334155;}
.rs-prob.on {color:#0F172A; font-weight:700;}
.rs-prob.on b {color:#0F172A; font-weight:750;}
.rs-track {height:10px; background:#EEF2F6; border-radius:999px; overflow:hidden;}
.rs-track div {height:100%; border-radius:999px;}

.rs-tiles {display:grid; gap:8px; margin:6px 0 10px;}
.rs-tile {background:#fff; border:1px solid #E2E8F0; border-radius:12px; padding:10px 12px;}
.rs-tile .v {font-size:1.3rem; font-weight:750; color:#0F172A; line-height:1.2;}
.rs-tile .l {font-size:.76rem; color:#64748B; margin-top:2px;}

.rs-dist {display:flex; height:16px; border-radius:999px; overflow:hidden; margin:4px 0 8px; background:#EEF2F6;}
.rs-legend span {display:inline-flex; align-items:center; gap:6px; margin:0 14px 4px 0; font-size:.82rem; color:#475569;}
.rs-legend i {width:10px; height:10px; border-radius:3px; display:inline-block;}

.rs-context {background:#F0FDFA; border:1px solid #99F6E4; color:#115E59; border-radius:12px; padding:10px 14px; font-size:.92rem;}
.rs-context.muted {background:#F8FAFC; border-color:#E2E8F0; color:#475569;}
.rs-footer {text-align:center; color:#94A3B8; font-size:.8rem; margin-top:28px;}

/* responsive rows: the two halves sit side by side on wide screens and stack on narrow ones */
.st-key-pick_row [data-testid="stHorizontalBlock"], .st-key-result_row [data-testid="stHorizontalBlock"] {flex-wrap:wrap;}
.st-key-pick_row [data-testid="stColumn"], .st-key-result_row [data-testid="stColumn"] {min-width:min(300px,100%);}
.st-key-gallery [data-testid="stHorizontalBlock"], .st-key-images [data-testid="stHorizontalBlock"],
.st-key-stages [data-testid="stHorizontalBlock"] {flex-wrap:nowrap;}
.st-key-gallery [data-testid="stColumn"], .st-key-images [data-testid="stColumn"],
.st-key-stages [data-testid="stColumn"] {min-width:0;}
.st-key-gallery button {min-height:30px; padding:0 4px;}
.st-key-gallery button p {font-size:.78rem; white-space:nowrap;}
.st-key-gallery img, .st-key-photo img {border-radius:12px;}
@media (max-width: 900px) {.rs-prob {grid-template-columns:110px 1fr 50px;} .rs-logo {display:none;}}
</style>
"""


def html(markup: str) -> None:
    """Render app-generated HTML. Markup must not contain blank lines (Markdown would end the block)."""
    st.markdown(markup, unsafe_allow_html=True)


def chip(text: str, fg: str, bg: str) -> str:
    return f'<span class="rs-chip" style="color:{fg};background:{bg}">{text}</span>'


def tiles(items, cols: int = 2) -> str:
    cells = "".join(f'<div class="rs-tile"><div class="v">{v}</div><div class="l">{label}</div></div>'
                    for label, v in items)
    # two tiles per row are kept fixed; wider rows wrap to fewer columns when space is short
    grid = "repeat(2,minmax(0,1fr))" if cols == 2 else f"repeat(auto-fit,minmax({110 if cols == 3 else 150}px,1fr))"
    return f'<div class="rs-tiles" style="grid-template-columns:{grid}">{cells}</div>'


def step_header(n, text: str) -> None:
    html(f'<div class="rs-step-h"><span>{n}</span>{text}</div>')


def severity_scale(active: int) -> str:
    cells = "".join(
        f'<div class="rs-seg{" on" if i == active else ""}"><div class="bar" style="background:{GRADE_COLOURS[i]}"></div>'
        f'<div class="lbl">{i} · {C.CLASS_LABELS[i]}</div></div>' for i in range(5))
    return f'<div class="rs-scale">{cells}</div>'


def result_card(res: dict) -> str:
    """Headline result for a non-specialist: stage, urgency, what to do next, place on the scale."""
    g, conf = res["grade"], res["confidence"]
    s, col = STAGE_INFO[g], GRADE_COLOURS[g]
    parts = [f'<div class="rs-result" style="--g:{col}">',
             '<div class="rs-kicker">Result</div>',
             f'<div class="rs-grade"><em>Grade {g}</em> · {s["name"]}</div>',
             chip(s["urgency"], col, f"{col}1F") + chip(f"{conf:.0%} confidence", "#334155", "#EEF2F6"),
             f'<div class="rs-advice"><b>What to do next:</b> {s["advice"].capitalize()}</div>']
    if conf < LOW_CONFIDENCE:
        parts.append('<div class="rs-warn">⚠ The AI is unsure about this photograph. Please ask an eye-care '
                     'professional to look at it.</div>')
    parts.append(severity_scale(g))
    parts.append("</div>")
    return "".join(parts)


def prob_bars(probs, active: int) -> str:
    return "".join(
        f'<div class="rs-prob{" on" if i == active else ""}"><span>{i} · {C.CLASS_LABELS[i]}</span>'
        f'<div class="rs-track"><div style="width:{max(float(p) * 100, 0.8):.1f}%;background:{GRADE_COLOURS[i]}"></div></div>'
        f'<b>{p:.1%}</b></div>' for i, p in enumerate(probs))


def grade_legend() -> str:
    return '<div class="rs-legend">' + "".join(
        f'<span><i style="background:{GRADE_COLOURS[g]}"></i>{g} · {C.CLASS_LABELS[g]}</span>' for g in range(5)) + "</div>"


# --------------------------------------------------------------------------- #
# Page frame
# --------------------------------------------------------------------------- #
html(CSS)
model, meta = load_model_and_meta()
ss = st.session_state
samples = (sorted(SAMPLES_DIR.glob("*.png")) + sorted(SAMPLES_DIR.glob("*.jpg"))) if SAMPLES_DIR.exists() else []

html('<div class="rs-hero"><div class="rs-logo">👁️</div><div>'
     '<div class="rs-title">RetinaScreen</div>'
     '<div class="rs-sub">Check a retina photograph for signs of diabetic retinopathy</div></div></div>')

if model is None:
    st.error("No trained model found in `models/`. Run `scripts/train.py --save-best` first.")
    st.stop()

tab_single, tab_batch, tab_bot, tab_card = st.tabs(["🔍 Screen a photo", "📂 Batch screening", "💬 Ask RetinaBot",
                                                    "ℹ️ About the model"])

# ---------------------------------------------------------------------- #
# Tab 1 - single image: choose a photo, then see the result.
# Everything technical sits in collapsed sections under the result.
# ---------------------------------------------------------------------- #
ss.setdefault("photo", None)        # (file name, encoded bytes) of the photograph being screened
ss.setdefault("upload_key", 0)


def store_upload() -> None:
    f = ss.get(f"upload_{ss['upload_key']}")
    if f is not None:
        ss["photo"] = (f.name, f.getvalue())


def choose_sample(name: str) -> None:
    ss["photo"] = (name, (SAMPLES_DIR / name).read_bytes())


def clear_photo() -> None:
    ss["photo"] = None
    ss["upload_key"] += 1            # a new key gives an empty uploader
    ss.pop("last_result", None)


with tab_single:
    if ss["photo"] is None:
        with st.container(key="pick_row"):
            up_col, how_col = st.columns([3, 2], gap="medium")
            with up_col:
                with st.container(border=True):
                    step_header(1, "Add a retina photograph")
                    st.file_uploader("Colour fundus photograph (PNG / JPG)", type=["png", "jpg", "jpeg"],
                                     key=f"upload_{ss['upload_key']}", on_change=store_upload)
                    if samples:
                        with st.expander("No photograph to hand? Try one of ours"):
                            html('<div class="rs-muted">Real photographs from the test data. The badge is the grade '
                                 'given by the eye specialists who labelled them.</div>')
                            with st.container(key="gallery"):
                                for i in range(0, len(samples), 5):
                                    cols = st.columns(5, gap="small")
                                    for col, p in zip(cols, samples[i:i + 5]):
                                        with col:
                                            st.image(thumbnail(str(p), sample_info(p.name)[1]), width="stretch")
                                            st.button("Use", key=f"pick_{p.name}", width="stretch", help=p.name,
                                                      on_click=choose_sample, args=(p.name,))
            with how_col:
                html('<div class="rs-how">'
                     '<div><i>📷</i><p><b>Add a photograph</b><span>A colour photo of the back of the eye.</span></p></div>'
                     '<div><i>🧠</i><p><b>The AI grades it</b><span>On the 5-stage international scale, in seconds.</span></p></div>'
                     '<div><i>✅</i><p><b>See what to do next</b><span>Clear advice on whether and how soon to see an '
                     'eye specialist.</span></p></div></div>')
    else:
        name, data = ss["photo"]
        with st.spinner("Checking the photograph ..."):
            res = grade_bytes(data, meta.get("run_name", ""), model, meta)
        ss["last_result"] = res

        bar_l, bar_r = st.columns([4, 1], vertical_alignment="center")
        bar_l.markdown(f'<div class="rs-photo-bar">📷 <b>{escape(name)}</b></div>', unsafe_allow_html=True)
        bar_r.button("Choose another photo", on_click=clear_photo, width="stretch")

        with st.container(key="result_row"):
            photo_col, res_col = st.columns([2, 3], gap="medium")
            with photo_col:
                with st.container(key="photo"):
                    st.image(res["rgb"], caption="Your photograph", width="stretch")
            with res_col:
                html(result_card(res))
                with st.container(horizontal=True, vertical_alignment="center"):
                    st.download_button("⬇ Download summary", screening_summary(name, res, meta),
                                       "screening_summary.txt", "text/plain", type="primary")
                    st.caption("Questions? Ask **RetinaBot** in the next tab.")

        with st.expander("🔥 Why this result? See where the AI looked"):
            with st.container(key="images"):
                c1, c2, c3 = st.columns([2, 2, 3], gap="medium")
                c1.image(res["processed"], caption="What the AI saw", width="stretch")
                slot = c2.empty()
                with c3:
                    st.caption("The heat-map (Grad-CAM) colours the parts of the photograph that pushed the AI "
                               "towards its answer: red mattered most, blue least. In diseased eyes these are "
                               "usually areas with lesions.")
                    alpha = st.slider("Heat-map strength", 0.2, 0.7, 0.4, 0.05)
                slot.image(overlay_heatmap(res["processed"], res["heatmap"], alpha),
                           caption="Where it looked", width="stretch")
        with st.expander("📊 How sure is the AI? Probability of each stage"):
            html(prob_bars(res["probs"], res["grade"]))
        with st.expander("⚙️ Technical details"):
            src_label, ref = sample_info(name)
            if ref is not None:
                agree = "agrees" if ref == res["grade"] else ("is one grade apart" if abs(ref - res["grade"]) == 1
                                                              else "disagrees")
                st.markdown(f"**Reference grade** in the {src_label}: {ref} - {C.CLASS_LABELS[ref]} "
                            f"(the prediction {agree}).")
            st.markdown(f"**Model:** {meta['backbone']}, {meta['preprocess']} preprocessing, {meta['img_size']} px input"
                        + (", test-time augmentation (4 flipped views)" if meta.get("tta") else "")
                        + f"  \n**Original size:** {res['rgb'].shape[1]} x {res['rgb'].shape[0]} px"
                        + f"  \n**Grading time:** {res['ms']:.0f} ms")
            st.markdown("**Preprocessing applied to this photograph** (identical to training)")
            stages = pipeline_stages(res["rgb"], meta)
            with st.container(key="stages"):
                cols = st.columns(len(stages))
                for col, (stage_name, img) in zip(cols, stages.items()):
                    col.image(img, caption=stage_name, width="stretch", clamp=True)

# ---------------------------------------------------------------------- #
# Tab 2 - batch
# ---------------------------------------------------------------------- #
ss.setdefault("batch_samples", False)
ss.setdefault("batch_key", 0)


def use_batch_samples() -> None:
    ss["batch_samples"] = True
    ss["batch_key"] += 1


def clear_batch() -> None:
    ss["batch_samples"] = False
    ss["batch_key"] += 1


with tab_batch:
    with st.container(border=True):
        step_header(1, "Screen many photographs at once")
        st.caption("For clinics: results are sorted with the most urgent eyes first and can be downloaded as a CSV file.")
        files = st.file_uploader("Fundus photographs (PNG / JPG)", type=["png", "jpg", "jpeg"],
                                 accept_multiple_files=True, key=f"batch_{ss['batch_key']}")
        if files:
            ss["batch_samples"] = False
        with st.container(horizontal=True):
            if samples:
                st.button(f"Try it with our {len(samples)} sample photographs", on_click=use_batch_samples)
            if files or ss["batch_samples"]:
                st.button("Clear", type="tertiary", on_click=clear_batch)

    items = ([(f.name, f.getvalue()) for f in files] if files
             else [(p.name, p.read_bytes()) for p in samples] if ss["batch_samples"] else [])
    if items:
        rows, thumbs = [], []
        progress = st.progress(0.0, text="Checking ...")
        for i, (name, data) in enumerate(items):
            res = grade_bytes(data, meta.get("run_name", ""), model, meta)
            _, ref = sample_info(name)
            rows.append({
                "image": name, "grade": res["grade"], "stage": C.CLASS_LABELS[res["grade"]],
                "confidence": round(res["confidence"], 3),
                "referable": res["grade"] >= C.REFERABLE_THRESHOLD,
                "urgency": STAGE_INFO[res["grade"]]["urgency"],
                "reference grade": ref,
                **{f"p_{C.CLASS_NAMES[k]}": round(float(res["probs"][k]), 3) for k in range(5)},
            })
            thumbs.append((name, res["overlay"], res["grade"], res["confidence"]))
            progress.progress((i + 1) / len(items), text=f"Checked {i + 1} of {len(items)}")
        progress.empty()
        df = pd.DataFrame(rows).sort_values(["grade", "confidence"], ascending=[False, False])

        n, n_ref = len(df), int(df["referable"].sum())
        html(tiles([("photographs checked", n),
                    ("need referral", f"{n_ref} <span class='rs-muted'>({n_ref / n:.0%})</span>"),
                    ("urgent", int((df["grade"] >= 3).sum())),
                    ("unsure: ask a specialist", int((df["confidence"] < LOW_CONFIDENCE).sum()))], cols=4))
        view = df[["image", "grade", "stage", "urgency", "confidence"]].copy()
        view["confidence"] = view["confidence"] * 100
        st.dataframe(view, hide_index=True, width="stretch", column_config={
            "image": st.column_config.TextColumn("Photograph", width="large"),
            "grade": st.column_config.NumberColumn("Grade", format="%d"),
            "stage": st.column_config.TextColumn("Stage"),
            "urgency": st.column_config.TextColumn("Urgency"),
            "confidence": st.column_config.ProgressColumn("Confidence", format="%.0f%%", min_value=0, max_value=100),
        })
        st.download_button("⬇ Download CSV report", df.to_csv(index=False).encode(), "dr_batch_report.csv",
                           "text/csv", type="primary")

        with st.expander("📊 Grade distribution"):
            counts = df["grade"].value_counts().reindex(range(5), fill_value=0)
            html('<div class="rs-dist">' + "".join(
                f'<div title="{C.CLASS_LABELS[g]}: {c}" style="width:{100 * c / n:.2f}%;background:{GRADE_COLOURS[g]}"></div>'
                for g, c in counts.items() if c) + "</div>" + grade_legend())
            known = df["reference grade"].notna()
            if known.any():
                k = df[known]
                exact = int((k["grade"] == k["reference grade"]).sum())
                near = int(((k["grade"] - k["reference grade"]).abs() <= 1).sum())
                st.caption(f"Agreement with the dataset's reference grade: {exact} of {len(k)} exact, "
                           f"{near} of {len(k)} within one grade.")
        with st.expander("🔥 Heat-maps of every photograph"):
            cols = st.columns(min(5, len(thumbs)))
            for i, (name, img, g, conf) in enumerate(thumbs):
                cols[i % len(cols)].image(img, caption=f"Grade {g} ({conf:.0%}) · {name[:22]}", width="stretch")

# ---------------------------------------------------------------------- #
# Tab 3 - chatbot
# ---------------------------------------------------------------------- #
with tab_bot:
    bot = RetinaBot()
    if "chat" not in ss:
        ss["chat"] = [("assistant", bot.greeting(ChatContext()))]
    last = ss.get("last_result")
    ctx = ChatContext(
        grade=last["grade"] if last else None,
        confidence=last["confidence"] if last else None,
        probabilities=[float(p) for p in last["probs"]] if last else None,
        model_name=meta["backbone"],
        metrics=meta.get("metrics", {}),
        external_metrics=meta.get("external_metrics", {}),
    )
    h1, h2 = st.columns([5, 1], vertical_alignment="center")
    with h1:
        if last:
            html(f'<div class="rs-context">💬 Talking about your result: <b>Grade {last["grade"]} · '
                 f'{STAGE_INFO[last["grade"]]["name"]}</b></div>')
        else:
            html('<div class="rs-context muted">Screen a photograph first to ask about your result, '
                 'or ask anything about diabetic retinopathy.</div>')
    if h2.button("Clear chat", type="tertiary"):
        ss["chat"] = [("assistant", bot.greeting(ChatContext()))]
        st.rerun()
    for role, text in ss["chat"]:
        with st.chat_message(role, avatar="👁️" if role == "assistant" else None):
            st.markdown(text)
    prompts = ["What does my result mean?", "What should I do next?", "What are the DR stages?", "What is the heat-map?"]
    clicked = None
    with st.container(horizontal=True):           # wraps onto several lines on narrow screens
        for p in prompts:
            if st.button(p):
                clicked = p
    user_msg = st.chat_input("Ask a question ...") or clicked
    if user_msg:
        ss["chat"].append(("user", user_msg))
        ss["chat"].append(("assistant", bot.reply(user_msg, ctx)))
        st.rerun()

# ---------------------------------------------------------------------- #
# Tab 4 - about the model (model card); details in collapsed sections
# ---------------------------------------------------------------------- #
with tab_card:
    tc = meta.get("training_config", {})
    ds = meta.get("datasets", {})
    m, ext = meta.get("metrics", {}), meta.get("external_metrics", {})
    keys = [("Accuracy", "accuracy", "{:.1%}"),
            ("Eyes needing referral that it finds", "referable_sensitivity", "{:.1%}"),
            ("Healthy eyes it correctly clears", "referable_specificity", "{:.1%}"),
            ("Agreement with specialists (kappa)", "qwk", "{:.2f}")]
    st.markdown("#### How accurate is it?")
    c1, c2 = st.columns(2, gap="large")
    for col, title, mm in [(c1, "On new photos from the same hospital (APTOS 2019)", m),
                           (c2, "On photos from another country (EyePACS 2015)", ext)]:
        if mm:
            col.markdown(f"**{title}**")
            with col:
                html(tiles([(label, fmt.format(metric(mm, key))) for label, key, fmt in keys], cols=2))
    st.caption("Accuracy drops on photographs from other cameras and populations, so a clinic would need to check "
               "and adapt the model with its own photographs before use. RetinaScreen is a coursework prototype, "
               "not a medical device.")

    with st.expander("🧠 How the model was built"):
        st.markdown(f"""
**Task** - 5-class diabetic retinopathy grading (ICDR scale) from colour fundus photographs, with a referable-DR
decision (grade 2 or worse) and referral urgency.

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
        hw = meta.get("trained_on", {})
        if hw:
            gpu = hw.get("gpu_name") or hw.get("gpu") or "CPU"
            gpu = " x ".join(gpu) if isinstance(gpu, list) else gpu
            st.caption(f"Trained with TensorFlow {hw.get('tensorflow')} on {gpu}.")
    with st.expander("⚖️ Intended use, limitations and ethics"):
        st.markdown("""
**Intended use** - decision *support* for screening programmes: prioritise which patients an
ophthalmologist should see first.  Not a diagnostic device.

**Limitations** - developed on one source (Aravind Eye Hospital, India) and validated externally
on EyePACS (USA); cannot detect diabetic macular oedema or non-DR pathology; adjacent grades are
frequently confused; image quality strongly affects results; model confidence is not clinical certainty.

**Ethics** - both datasets are anonymised and released under CC0; predictions are explained with
Grad-CAM so a clinician can verify that the evidence is anatomically plausible.
""")
    with st.expander("📈 All metrics"):
        table = pd.DataFrame({"APTOS 2019 test (internal)": pd.Series(m)})
        if ext:
            table["EyePACS 2015 (external)"] = pd.Series(ext)
        st.table(table.apply(lambda col: col.map(lambda v: f"{v:.4f}")))

    # Training and evaluation evidence exported by the Kaggle notebook (figures/ folder).
    evidence = [
        ("Experiments", [(FIGURES_DIR / "09_experiment_results.png", "Experiments on the validation subset (one factor per stage)")]),
        ("Training curves", [(FIGURES_DIR / "09_curves_final.png", "Accuracy and loss curves of the final model")]),
        ("Confusion matrices", [(FIGURES_DIR / "10_confusion_matrices.png", "Confusion matrices on the internal test subset")]),
        ("Per-grade metrics", [(FIGURES_DIR / "10_per_class_metrics_and_roc.png", "Per-grade precision, recall, F1 and ROC curves")]),
        ("Calibration", [(FIGURES_DIR / "10_confidence_calibration.png", "Confidence calibration")]),
        ("Grad-CAM", [(FIGURES_DIR / "10_gradcam_test.png", "Grad-CAM on test images")]),
        ("External validation", [(FIGURES_DIR / "11_internal_vs_external.png", "Internal test vs external validation"),
                                 (FIGURES_DIR / "11_external_confusion_matrices.png", "External validation confusion matrices")]),
    ]
    evidence = [(t, [(p, cap) for p, cap in figs if p.exists()]) for t, figs in evidence]
    evidence = [(t, figs) for t, figs in evidence if figs]
    if evidence:
        with st.expander("🖼️ Training and evaluation evidence (figures)"):
            for tab, (_, figs) in zip(st.tabs([t for t, _ in evidence]), evidence):
                with tab:
                    for p, cap in figs:
                        st.image(str(p), caption=cap, width="stretch")

html('<div class="rs-footer">RetinaScreen · university coursework prototype · not a medical device</div>')

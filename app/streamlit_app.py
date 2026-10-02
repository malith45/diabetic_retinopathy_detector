"""
RetinaScreen - Diabetic Retinopathy stage detection demo (Streamlit).

Run locally:      streamlit run app/streamlit_app.py
Requires:         models/best_model.keras and models/model_metadata.json
                  (exported by the Kaggle notebook, or by scripts/train.py --save-best)

Tabs
----
1. Screen an image  - upload a fundus photo or pick a real test photo -> stage on the
                      ICDR scale, probabilities, Grad-CAM heat-map, referral advice,
                      preprocessing preview and a downloadable summary
2. Batch screening  - grade many images at once, summary figures and a CSV report
3. RetinaBot        - rule-based assistant that explains results and DR stages
4. Model card       - dataset, architecture, metrics, evidence and limitations
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
                   initial_sidebar_state="expanded")


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
[data-testid="stMainBlockContainer"], .block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1380px;}
[data-testid="stHeader"] {background: transparent;}
.stTabs [data-baseweb="tab-list"] {gap: 4px;}
.stTabs [data-baseweb="tab"] {height: 46px; padding: 0 18px;}
.stTabs [data-baseweb="tab"] p {font-size: 0.98rem; font-weight: 600;}

.rs-hero {display:flex; align-items:center; gap:18px; padding:16px 24px; margin:0 0 6px; border-radius:16px; color:#fff;
  background:linear-gradient(120deg,#0F766E 0%,#115E59 58%,#164E63 100%); box-shadow:0 8px 24px rgba(15,118,110,.18);}
.rs-logo {font-size:2.3rem; line-height:1; background:rgba(255,255,255,.14); border-radius:14px; padding:10px 12px;}
.rs-title {font-size:1.75rem; font-weight:750; letter-spacing:-.02em; line-height:1.15;}
.rs-sub {color:#CCFBF1; margin:2px 0 10px; font-size:1rem;}
.rs-pill {display:inline-block; margin:0 6px 4px 0; padding:3px 11px; border-radius:999px; font-size:.78rem;
  background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.28); color:#fff;}

.rs-step-h {display:flex; align-items:center; gap:10px; font-weight:700; font-size:1.02rem; color:#0F172A; margin:2px 0 10px;}
.rs-step-h span {display:inline-flex; align-items:center; justify-content:center; width:24px; height:24px; border-radius:999px;
  background:#0F766E; color:#fff; font-size:.8rem; flex:none;}
.rs-or {text-align:center; color:#64748B; font-size:.76rem; margin:12px 0 0; text-transform:uppercase; letter-spacing:.09em;}
.rs-gallery-h {font-size:.8rem; font-weight:700; color:#334155; margin:12px 0 2px;}
/* responsive rows: the two halves sit side by side on wide screens and stack on narrow ones */
.st-key-pick_row [data-testid="stHorizontalBlock"], .st-key-result_row [data-testid="stHorizontalBlock"] {flex-wrap:wrap;}
.st-key-pick_row [data-testid="stColumn"], .st-key-result_row [data-testid="stColumn"] {min-width:min(330px,100%);}
.st-key-gallery [data-testid="stHorizontalBlock"], .st-key-images [data-testid="stHorizontalBlock"],
.st-key-actions [data-testid="stHorizontalBlock"], .st-key-stages [data-testid="stHorizontalBlock"] {flex-wrap:nowrap;}
.st-key-gallery [data-testid="stColumn"], .st-key-images [data-testid="stColumn"],
.st-key-actions [data-testid="stColumn"], .st-key-stages [data-testid="stColumn"] {min-width:0;}
.st-key-gallery button {min-height:30px; padding:0 4px;}
.st-key-gallery button p {font-size:.78rem; white-space:nowrap;}
.st-key-gallery img {border-radius:10px;}
.rs-muted {color:#64748B; font-size:.86rem;}

.rs-result {background:#fff; border:1px solid #E2E8F0; border-left:7px solid var(--g); border-radius:14px;
  padding:18px 22px 16px; margin-bottom:14px; box-shadow:0 1px 3px rgba(15,23,42,.05);}
.rs-kicker {font-size:.72rem; font-weight:700; letter-spacing:.09em; text-transform:uppercase; color:#64748B;}
.rs-grade {font-size:1.65rem; font-weight:750; color:#0F172A; letter-spacing:-.01em; margin:2px 0 10px; line-height:1.25;}
.rs-grade em {font-style:normal; color:var(--g);}
.rs-chip {display:inline-block; padding:3px 11px; margin:0 6px 6px 0; border-radius:999px; font-size:.8rem; font-weight:650;}
.rs-advice {color:#334155; margin:6px 0 2px; font-size:.95rem;}
.rs-note {font-size:.84rem; color:#475569; margin-top:8px;}
.rs-warn {background:#FFF7ED; border:1px solid #FED7AA; color:#9A3412; border-radius:10px; padding:7px 11px; font-size:.85rem; margin-top:10px;}
.rs-scale {display:grid; grid-template-columns:repeat(5,1fr); gap:6px; margin-top:16px; align-items:end;}
.rs-seg .bar {height:8px; border-radius:6px; opacity:.25;}
.rs-seg.on .bar {height:14px; opacity:1;}
.rs-seg .lbl {font-size:.72rem; color:#94A3B8; text-align:center; margin-top:5px; line-height:1.2;}
.rs-seg.on .lbl {color:#0F172A; font-weight:700;}

.rs-prob {display:grid; grid-template-columns:150px 1fr 56px; align-items:center; gap:12px; margin:8px 0; font-size:.9rem; color:#475569;}
.rs-prob b {text-align:right; font-weight:600; color:#334155;}
.rs-prob.on {color:#0F172A; font-weight:700;}
.rs-prob.on b {color:#0F172A; font-weight:750;}
.rs-track {height:10px; background:#EEF2F6; border-radius:999px; overflow:hidden;}
.rs-track div {height:100%; border-radius:999px;}

.rs-tiles {display:grid; gap:8px; margin:6px 0 10px;}
.rs-tile {background:#fff; border:1px solid #E2E8F0; border-radius:12px; padding:10px 12px;}
.rs-tile .v {font-size:1.3rem; font-weight:750; color:#0F172A; line-height:1.2;}
.rs-tile .l {font-size:.74rem; color:#64748B; margin-top:2px;}

.rs-empty {background:#fff; border:1px dashed #CBD5E1; border-radius:14px; padding:22px; margin-bottom:12px; text-align:center; color:#475569;}
.rs-empty b {color:#0F172A; font-size:1.1rem;}
.rs-steps {display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px;}
.rs-stepcard {background:#fff; border:1px solid #E2E8F0; border-radius:14px; padding:16px;}
.rs-stepcard .i {font-size:1.5rem;}
.rs-stepcard .t {font-weight:700; color:#0F172A; margin:6px 0 4px;}
.rs-stepcard .d {font-size:.86rem; color:#475569;}

.rs-dist {display:flex; height:16px; border-radius:999px; overflow:hidden; margin:4px 0 8px; background:#EEF2F6;}
.rs-legend span {display:inline-flex; align-items:center; gap:6px; margin:0 14px 4px 0; font-size:.82rem; color:#475569;}
.rs-legend i {width:10px; height:10px; border-radius:3px; display:inline-block;}

.rs-context {background:#F0FDFA; border:1px solid #99F6E4; color:#115E59; border-radius:12px; padding:10px 14px; font-size:.92rem;}
.rs-context.muted {background:#F8FAFC; border-color:#E2E8F0; color:#475569;}

.rs-side-brand {font-size:1.25rem; font-weight:750; color:#0F172A;}
.rs-side-h {font-size:.74rem; font-weight:700; color:#334155; margin:16px 0 2px; text-transform:uppercase; letter-spacing:.06em;}
.rs-side-model {font-size:.84rem; color:#475569; background:#F1F5F9; border-radius:10px; padding:8px 10px; margin-top:10px;}
.rs-card {background:#fff; border:1px solid #E2E8F0; border-radius:14px; padding:16px 18px; margin-bottom:12px;}
.rs-card-h {font-weight:700; color:#0F172A; margin-bottom:8px;}
@media (max-width: 900px) {.rs-prob {grid-template-columns:110px 1fr 50px;} .rs-hero {padding:16px;} .rs-logo {display:none;}}
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
    # two tiles per row are kept fixed (sidebar); wider rows wrap to fewer columns when space is short
    grid = "repeat(2,minmax(0,1fr))" if cols == 2 else f"repeat(auto-fit,minmax({110 if cols == 3 else 150}px,1fr))"
    return f'<div class="rs-tiles" style="grid-template-columns:{grid}">{cells}</div>'


def step_header(n, text: str) -> None:
    html(f'<div class="rs-step-h"><span>{n}</span>{text}</div>')


def severity_scale(active: int) -> str:
    cells = "".join(
        f'<div class="rs-seg{" on" if i == active else ""}"><div class="bar" style="background:{GRADE_COLOURS[i]}"></div>'
        f'<div class="lbl">{i} · {C.CLASS_LABELS[i]}</div></div>' for i in range(5))
    return f'<div class="rs-scale">{cells}</div>'


def result_card(res: dict, reference=None, source=None) -> str:
    """Headline result: stage, confidence, urgency, advice and position on the ICDR scale."""
    g, conf = res["grade"], res["confidence"]
    s, col = STAGE_INFO[g], GRADE_COLOURS[g]
    referable = g >= C.REFERABLE_THRESHOLD
    chips = (chip(f"{conf:.0%} confidence", "#0F172A", "#EEF2F6")
             + chip(f"Urgency: {s['urgency']}", col, f"{col}1F")
             + chip("Referable DR (grade 2 or worse)" if referable else "Not referable",
                    "#9F1239" if referable else "#166534", "#FFE4E6" if referable else "#DCFCE7"))
    parts = [f'<div class="rs-result" style="--g:{col}">',
             '<div class="rs-kicker">Predicted stage (ICDR scale)</div>',
             f'<div class="rs-grade"><em>Grade {g}</em> · {s["name"]}</div>',
             chips,
             f'<div class="rs-advice"><b>Recommended action:</b> {s["advice"].capitalize()}</div>']
    if reference is not None:
        agree = "agrees" if reference == g else ("one grade apart" if abs(reference - g) == 1 else "disagrees")
        mark = "✓" if reference == g else "≠"
        parts.append(f'<div class="rs-note">{mark} Reference grade in the {source}: <b>{reference} · '
                     f'{C.CLASS_LABELS[reference]}</b> (prediction {agree})</div>')
    if conf < LOW_CONFIDENCE:
        parts.append('<div class="rs-warn">⚠ Low confidence: the photograph lies between two stages and should be '
                     'reviewed by a human grader.</div>')
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
# Page frame: theme additions, header and sidebar
# --------------------------------------------------------------------------- #
html(CSS)
model, meta = load_model_and_meta()
ss = st.session_state
samples = (sorted(SAMPLES_DIR.glob("*.png")) + sorted(SAMPLES_DIR.glob("*.jpg"))) if SAMPLES_DIR.exists() else []

pills = "".join(f'<span class="rs-pill">{p}</span>' for p in [
    f"{meta['backbone'] if meta else 'CNN'} · transfer learning", "5 ICDR stages + referral advice",
    "Grad-CAM explanations", "External validation on EyePACS 2015", "RetinaBot assistant"])
html('<div class="rs-hero"><div class="rs-logo">👁️</div><div>'
     '<div class="rs-title">RetinaScreen</div>'
     '<div class="rs-sub">AI-assisted diabetic retinopathy grading from a single colour fundus photograph</div>'
     f'{pills}</div></div>')

with st.sidebar:
    html('<div class="rs-side-brand">👁️ RetinaScreen</div><div class="rs-muted">How good is the model?</div>')
    if meta:
        m, ext = meta.get("metrics", {}), meta.get("external_metrics", {})
        html(f'<div class="rs-side-model"><b>{meta["backbone"]}</b> · {meta["preprocess"]} preprocessing · '
             f'{meta["img_size"]} px{" · test-time augmentation" if meta.get("tta") else ""}</div>')
        if m:
            html('<div class="rs-side-h">Internal test · APTOS 2019</div>' + tiles([
                ("Accuracy", f"{metric(m, 'accuracy'):.1%}"), ("Quadratic kappa", f"{metric(m, 'qwk'):.3f}"),
                ("Macro F1", f"{metric(m, 'macro_f1'):.3f}"), ("Referable sensitivity", f"{metric(m, 'referable_sensitivity'):.1%}")]))
        if ext:
            html('<div class="rs-side-h">External validation · EyePACS 2015</div>' + tiles([
                ("Accuracy", f"{metric(ext, 'accuracy'):.1%}"), ("Quadratic kappa", f"{metric(ext, 'qwk'):.3f}"),
                ("Macro F1", f"{metric(ext, 'macro_f1'):.3f}"), ("Referable sensitivity", f"{metric(ext, 'referable_sensitivity'):.1%}")]))
            st.caption("External = another country, other cameras and patients, never seen during training. "
                       "The drop is analysed in the report.")
    else:
        st.error("No trained model found in `models/`. Run `scripts/train.py --save-best` first.")
    html('<div class="rs-side-h">Grades (ICDR scale)</div>' + grade_legend())
    st.caption("Coursework prototype - not a medical device.")

tab_single, tab_batch, tab_bot, tab_card = st.tabs(["🔍 Screen an image", "📂 Batch screening", "💬 RetinaBot", "📋 Model card"])

# ---------------------------------------------------------------------- #
# Tab 1 - single image
# ---------------------------------------------------------------------- #
ss.setdefault("sample", None)
ss.setdefault("upload_key", 0)


def choose_sample(name: str) -> None:
    ss["sample"] = name
    ss["upload_key"] += 1            # a new key empties the uploader, so the sample is shown


def clear_selection() -> None:
    ss["sample"] = None
    ss["upload_key"] += 1
    ss.pop("last_result", None)


with tab_single:
    # Row 1: choose a photograph (uploader | gallery). Rows wrap to one column on narrow screens.
    with st.container(key="pick_row"):
        up_col, gal_col = st.columns([2, 3], gap="medium")
        with up_col:
            with st.container(border=True):
                step_header(1, "Upload a fundus photograph")
                file = st.file_uploader("Colour fundus photograph (PNG / JPG)", type=["png", "jpg", "jpeg"],
                                        key=f"upload_{ss['upload_key']}")
                if file is not None:
                    ss["sample"] = None
                html('<div class="rs-muted">The photograph is graded on this server and is not stored.</div>')
                if file is not None or ss["sample"]:
                    st.button("Clear and start again", type="tertiary", on_click=clear_selection)
        with gal_col:
            with st.container(border=True):
                step_header("or", "Try a real test photograph")
                html('<div class="rs-muted">The coloured badge is the grade given by the human graders of the dataset.</div>')
                groups = [("APTOS 2019 · held-out test set", [p for p in samples if p.name.lower().startswith("aptos")]),
                          ("EyePACS 2015 · external, never seen in training",
                           [p for p in samples if p.name.lower().startswith("eyepacs")])]
                groups = [(t, paths) for t, paths in groups if paths]
                if groups:
                    with st.container(key="gallery"):
                        for tab, (_, paths) in zip(st.tabs([t for t, _ in groups]), groups):
                            with tab:
                                for i in range(0, len(paths), 5):
                                    cols = st.columns(5, gap="small")
                                    for col, p in zip(cols, paths[i:i + 5]):
                                        _, ref = sample_info(p.name)
                                        with col:
                                            st.image(thumbnail(str(p), ref), width="stretch")
                                            chosen = ss["sample"] == p.name
                                            st.button("✓" if chosen else "Use", key=f"pick_{p.name}", width="stretch",
                                                      type="primary" if chosen else "secondary", help=p.name,
                                                      on_click=choose_sample, args=(p.name,))

    # Row 2: the result (headline + probabilities | images + preprocessing)
    source = file if file is not None else (SAMPLES_DIR / ss["sample"] if ss["sample"] else None)
    with st.container(key="result_row"):
        if model is None:
            st.error("Model not available: `models/best_model.keras` is missing.")
        elif source is None:
            steps = [("📷", "1 · Photograph", "Upload a colour fundus photograph, or pick one of the real test photographs."),
                     ("🧹", "2 · Preprocess", f"Black border cropped, padded to a square and resized to {meta['img_size']} px, "
                                             "exactly as during training."),
                     ("🧠", "3 · Grade", f"{meta['backbone']} (ImageNet transfer learning) scores the five ICDR stages."),
                     ("🔥", "4 · Explain and advise", "Grad-CAM shows where the network looked; the stage is mapped to "
                                                     "referral advice and urgency.")]
            html('<div class="rs-empty"><b>Choose a photograph to begin</b><br>'
                 'The result appears here within a few seconds. This is what happens to it:</div>'
                 '<div class="rs-steps">' + "".join(
                     f'<div class="rs-stepcard"><div class="i">{i}</div><div class="t">{t}</div><div class="d">{d}</div></div>'
                     for i, t, d in steps) + "</div>")
        else:
            name = file.name if file is not None else source.name
            data = file.getvalue() if file is not None else source.read_bytes()
            with st.spinner("Grading the photograph ..."):
                res = grade_bytes(data, meta.get("run_name", ""), model, meta)
            ss["last_result"] = res
            src_label, ref = sample_info(name)

            res_col, img_col = st.columns([1, 1], gap="medium")
            with res_col:
                html(result_card(res, ref, src_label))
                with st.container(border=True):
                    step_header(2, "Probability of each stage")
                    html(prob_bars(res["probs"], res["grade"]))
                    with st.container(key="actions"):
                        b1, b2 = st.columns([1, 1], vertical_alignment="center")
                        b1.download_button("⬇ Download screening summary", screening_summary(name, res, meta),
                                           "screening_summary.txt", "text/plain", type="primary", width="stretch")
                        b2.caption("💬 Questions? Open the **RetinaBot** tab: it knows this result.")
            with img_col:
                with st.container(border=True, key="images"):
                    step_header(3, "What the model saw and where it looked")
                    c1, c2 = st.columns(2)
                    c1.image(res["processed"], caption=f"Model input ({meta['preprocess']}, {meta['img_size']} px)",
                             width="stretch")
                    slot = c2.empty()
                    alpha = c2.slider("Heat-map opacity", 0.2, 0.7, 0.4, 0.05)
                    slot.image(overlay_heatmap(res["processed"], res["heatmap"], alpha),
                               caption="Grad-CAM: warm colours drove the prediction", width="stretch")
                with st.expander("Preprocessing pipeline applied to this photograph"):
                    stages = pipeline_stages(res["rgb"], meta)
                    with st.container(key="stages"):
                        cols = st.columns(len(stages))
                        for col, (stage_name, img) in zip(cols, stages.items()):
                            col.image(img, caption=stage_name, width="stretch", clamp=True)
                st.caption(f"{escape(name)} · original {res['rgb'].shape[1]} x {res['rgb'].shape[0]} px · graded in "
                           f"{res['ms']:.0f} ms" + (" with test-time augmentation (4 flipped views)" if meta.get("tta") else ""))

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
    top_l, top_r = st.columns([3, 2], gap="large")
    with top_l:
        with st.container(border=True):
            step_header(1, "Add photographs")
            files = st.file_uploader("Upload several fundus photographs", type=["png", "jpg", "jpeg"],
                                     accept_multiple_files=True, key=f"batch_{ss['batch_key']}")
            if files:
                ss["batch_samples"] = False
            with st.container(horizontal=True):
                if samples:
                    st.button(f"Grade all {len(samples)} sample photographs", on_click=use_batch_samples)
                if files or ss["batch_samples"]:
                    st.button("Clear", type="tertiary", on_click=clear_batch)
    with top_r:
        html('<div class="rs-card"><div class="rs-card-h">How a clinic would use this</div>'
             '<div class="rs-muted">Grade a whole session of photographs at once. Results are sorted with the most '
             'urgent eyes first, low-confidence images are counted for human review, and the CSV report can be '
             'imported into a patient-management system.</div></div>')

    items = ([(f.name, f.getvalue()) for f in files] if files
             else [(p.name, p.read_bytes()) for p in samples] if ss["batch_samples"] else [])
    if items and model is not None:
        rows, thumbs = [], []
        progress = st.progress(0.0, text="Grading ...")
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
            progress.progress((i + 1) / len(items), text=f"Graded {i + 1} of {len(items)}")
        progress.empty()
        df = pd.DataFrame(rows).sort_values(["grade", "confidence"], ascending=[False, False])

        step_header(2, "Results")
        n, n_ref = len(df), int(df["referable"].sum())
        html(tiles([("photographs graded", n),
                    ("referable (grade 2 or worse)", f"{n_ref} <span class='rs-muted'>({n_ref / n:.0%})</span>"),
                    ("urgent (grade 3 or 4)", int((df["grade"] >= 3).sum())),
                    ("low confidence: human review", int((df["confidence"] < LOW_CONFIDENCE).sum()))], cols=4))
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

        view = df[["image", "grade", "stage", "confidence", "referable", "urgency"] + (["reference grade"] if known.any() else [])].copy()
        view["confidence"] = view["confidence"] * 100
        st.dataframe(view, hide_index=True, width="stretch", column_config={
            "image": st.column_config.TextColumn("Image", width="large"),
            "grade": st.column_config.NumberColumn("Grade", format="%d"),
            "stage": st.column_config.TextColumn("Stage"),
            "confidence": st.column_config.ProgressColumn("Confidence", format="%.0f%%", min_value=0, max_value=100),
            "referable": st.column_config.CheckboxColumn("Referable"),
            "urgency": st.column_config.TextColumn("Urgency"),
            "reference grade": st.column_config.NumberColumn("Reference", format="%d"),
        })
        st.download_button("⬇ Download CSV report", df.to_csv(index=False).encode(), "dr_batch_report.csv",
                           "text/csv", type="primary")
        with st.expander("Grad-CAM heat-maps of every photograph", expanded=False):
            cols = st.columns(min(5, len(thumbs)))
            for i, (name, img, g, conf) in enumerate(thumbs):
                cols[i % len(cols)].image(img, caption=f"Grade {g} ({conf:.0%}) · {name[:22]}", width="stretch")
    elif not items:
        st.info("Upload photographs above, or grade the bundled sample photographs with one click.")

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
        model_name=meta["backbone"] if meta else "CNN",
        metrics=meta.get("metrics", {}) if meta else {},
        external_metrics=meta.get("external_metrics", {}) if meta else {},
    )
    h1, h2 = st.columns([5, 1], vertical_alignment="center")
    with h1:
        if last:
            html(f'<div class="rs-context">💬 Discussing your last result: <b>Grade {last["grade"]} · '
                 f'{STAGE_INFO[last["grade"]]["name"]}</b> ({last["confidence"]:.0%} confidence)</div>')
        else:
            html('<div class="rs-context muted">Tip: screen a photograph first, then ask RetinaBot about the result. '
                 'General questions about diabetic retinopathy work at any time.</div>')
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
    user_msg = st.chat_input("Ask about your result or about diabetic retinopathy ...") or clicked
    if user_msg:
        ss["chat"].append(("user", user_msg))
        ss["chat"].append(("assistant", bot.reply(user_msg, ctx)))
        st.rerun()

# ---------------------------------------------------------------------- #
# Tab 4 - model card
# ---------------------------------------------------------------------- #
with tab_card:
    if meta:
        tc = meta.get("training_config", {})
        ds = meta.get("datasets", {})
        m, ext = meta.get("metrics", {}), meta.get("external_metrics", {})
        keys = [("Accuracy", "accuracy", "{:.1%}"), ("Quadratic kappa", "qwk", "{:.3f}"), ("Macro F1", "macro_f1", "{:.3f}"),
                ("Referable sensitivity", "referable_sensitivity", "{:.1%}"),
                ("Referable specificity", "referable_specificity", "{:.1%}"), ("Referable AUC", "referable_auc", "{:.3f}")]
        c1, c2 = st.columns(2, gap="large")
        for col, title, mm in [(c1, "Internal test · APTOS 2019 (held-out 15 %)", m),
                               (c2, "External validation · EyePACS 2015 (never seen)", ext)]:
            if mm:
                col.markdown(f"**{title}**")
                with col:
                    html(tiles([(label, fmt.format(metric(mm, key))) for label, key, fmt in keys], cols=3))

        l, r = st.columns([3, 2], gap="large")
        with l:
            with st.container(border=True):
                st.markdown(f"""
#### What the model does
5-class diabetic retinopathy grading (ICDR scale) from colour fundus photographs, with a referable-DR
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
        with r:
            with st.container(border=True):
                st.markdown("""
#### Responsible use
**Intended use** - decision *support* for screening programmes: prioritise which patients an
ophthalmologist should see first.  Not a diagnostic device.

**Limitations** - developed on one source (Aravind Eye Hospital, India) and validated externally
on EyePACS (USA); cannot detect diabetic macular oedema or non-DR pathology; adjacent grades are
frequently confused; image quality strongly affects results; model confidence is not clinical certainty.

**Ethics** - both datasets are anonymised and released under CC0; predictions are explained with
Grad-CAM so a clinician can verify that the evidence is anatomically plausible.
""")
            hw = meta.get("trained_on", {})
            if hw:
                gpu = hw.get("gpu_name") or hw.get("gpu") or "CPU"
                gpu = " x ".join(gpu) if isinstance(gpu, list) else gpu
                st.caption(f"Trained with TensorFlow {hw.get('tensorflow')} on {gpu}.")
        with st.expander("All metrics"):
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
        st.markdown("#### Training and evaluation evidence")
        for tab, (_, figs) in zip(st.tabs([t for t, _ in evidence]), evidence):
            with tab:
                for p, cap in figs:
                    st.image(str(p), caption=cap, width="stretch")

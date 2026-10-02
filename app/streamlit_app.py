"""
RetinaScreen - Diabetic Retinopathy stage detection demo (Streamlit).

Run locally:      streamlit run app/streamlit_app.py
Requires:         models/best_model.keras and models/model_metadata.json
                  (exported by the Kaggle notebook, or by scripts/train.py --save-best)

Tabs
----
1. Screen a photo   - drop a fundus photo or tap a sample -> grade, stage, urgency and what
                      to do next; a switch fades the Grad-CAM heat-map over the photo;
                      probabilities and technical details under "More details"
2. Batch screening  - grade many images at once, most urgent first, with a CSV report
3. RetinaBot        - rule-based assistant that explains results and DR stages
4. About            - accuracy in plain language; model card, metrics and evidence figures
                      in collapsed sections
"""

from __future__ import annotations

import base64
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
FRIENDLY = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

# Small line icons (Lucide, ISC licence) used in the HTML parts of the page
ICON = {
    "eye": '<path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/>',
    "upload": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>',
    "pulse": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "check": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
    "next": '<path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>',
    "alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
}


def icon(name: str, size: int = 20, colour: str = "currentColor") -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{colour}" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round">{ICON[name]}</svg>')


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
body, .stApp, [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p, button p, input, textarea,
[data-testid="stWidgetLabel"] p, [data-testid="stCaptionContainer"] {font-family:'Plus Jakarta Sans', system-ui, -apple-system, 'Segoe UI', sans-serif !important;}
.stApp {background: radial-gradient(1100px 480px at 0% -10%, rgba(20,184,166,.13), transparent 60%),
                    radial-gradient(900px 420px at 100% -5%, rgba(14,165,233,.10), transparent 60%), #F7FAFC;}
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer {display:none !important;}
[data-testid="stMainBlockContainer"], .block-container {padding-top: 1.4rem; padding-bottom: 2.5rem; max-width: 1120px;}

/* top bar */
.rs-top {display:flex; align-items:center; justify-content:space-between; padding:0 2px 16px;}
.rs-brand {display:flex; align-items:center; gap:12px;}
.rs-mark {width:42px; height:42px; border-radius:13px; display:grid; place-items:center; color:#fff;
  background:linear-gradient(135deg,#14B8A6,#0F766E); box-shadow:0 6px 16px rgba(15,118,110,.30);}
.rs-brand b {display:block; font-size:1.15rem; color:#0F172A; letter-spacing:-.01em; line-height:1.2;}
.rs-brand small {color:#64748B; font-size:.8rem;}
.rs-proto {font-size:.74rem; color:#0F766E; background:#CCFBF1; padding:5px 11px; border-radius:999px; font-weight:700;}

/* pill navigation (react-aria tabs in Streamlit >= 1.5x, BaseWeb tabs in older versions) */
.stTabs [role="tablist"] {gap:4px; background:#fff; padding:5px; border-radius:999px; border:1px solid #E2E8F0 !important;
  box-shadow:0 1px 2px rgba(15,23,42,.05), 0 6px 18px rgba(15,23,42,.06); width:fit-content; max-width:100%; margin:0 auto 12px;
  overflow-x:auto;}
.stTabs [role="tab"] {height:40px; padding:0 18px; border-radius:999px; background:transparent; white-space:nowrap;
  display:flex; align-items:center; border:none !important; transition:background .15s;}
.stTabs [role="tab"]:hover {background:#F1F5F9;}
.stTabs [role="tab"] p {font-size:.93rem; font-weight:600; color:#475569; margin:0;}
.stTabs [role="tab"][aria-selected="true"] {background:#0F766E;}
.stTabs [role="tab"][aria-selected="true"] p, .stTabs [role="tab"][aria-selected="true"] span {color:#fff !important;}
.stTabs [role="tab"] > div:not([data-testid]), .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display:none !important;}

/* buttons */
.stButton button, .stDownloadButton button {border-radius:999px; font-weight:600; min-height:42px; padding:0 20px; transition:all .15s ease;}
[data-testid="stBaseButton-primary"] {box-shadow:0 6px 16px rgba(15,118,110,.25);}
[data-testid="stBaseButton-primary"]:hover {transform:translateY(-1px); box-shadow:0 10px 22px rgba(15,118,110,.30);}
[data-testid="stBaseButton-secondary"] {background:#fff; border:1px solid #E2E8F0;}
[data-testid="stBaseButton-secondary"]:hover {border-color:#0F766E; color:#0F766E;}

/* drag-and-drop area */
[data-testid="stFileUploaderDropzone"] {min-height:200px; display:flex; flex-direction:column; align-items:center; justify-content:center;
  gap:10px; padding:26px; background:#fff; border:2px dashed #99F6E4; border-radius:22px; transition:all .2s ease;
  box-shadow:0 1px 2px rgba(15,23,42,.04);}
[data-testid="stFileUploaderDropzone"]:hover {border-color:#0F766E; background:#F0FDFA;}
[data-testid="stFileUploaderDropzone"]::before {content:"Drag and drop a retina photo here"; display:block; padding-top:62px;
  font-weight:700; font-size:1.05rem; color:#0F172A; text-align:center;
  background:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='52' height='52' viewBox='0 0 24 24' fill='none' stroke='%230F766E' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4'/%3E%3Cpolyline points='17 8 12 3 7 8'/%3E%3Cline x1='12' y1='3' x2='12' y2='15'/%3E%3C/svg%3E") no-repeat top center;}
[data-testid="stFileUploaderDropzone"] {flex-direction:column !important; align-items:center !important; text-align:center;}
[data-testid="stFileUploaderDropzone"] > span, [data-testid="stFileUploaderDropzoneInstructions"] {align-self:center !important;}
.st-key-batch_top [data-testid="stFileUploaderDropzone"]::before {content:"Drag and drop retina photos here";}
[data-testid="stFileUploaderDropzoneInstructions"] span {font-size:0;}
[data-testid="stFileUploaderDropzoneInstructions"] span::after {content:"PNG or JPG · the photo is not stored"; font-size:.82rem; color:#64748B;}
[data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"] {background:#0F766E; color:#fff; border:none;
  box-shadow:0 6px 16px rgba(15,118,110,.25);}
[data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"] p,
[data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"] span {color:#fff;}

/* expanders */
[data-testid="stExpander"] details {border:1px solid #E2E8F0; border-radius:18px; background:#fff; box-shadow:0 1px 2px rgba(15,23,42,.04);}
[data-testid="stExpander"] summary p {font-weight:600;}

/* landing */
.st-key-landing, .st-key-batch_top {width:100%; max-width:760px; margin:0 auto;}
.rs-hero {text-align:center; padding:14px 0 18px;}
.rs-hero .t {font-size:2.35rem; line-height:1.15; font-weight:800; letter-spacing:-.035em; color:#0F172A; margin-bottom:10px;}
.rs-hero .t span {background:linear-gradient(90deg,#0F766E,#0EA5E9); -webkit-background-clip:text; background-clip:text; color:transparent;}
.rs-hero p {color:#475569; font-size:1.04rem; max-width:560px; margin:0 auto;}
.rs-divider {display:flex; align-items:center; gap:14px; color:#64748B; font-size:.86rem; font-weight:600; margin:22px 0 10px;}
.rs-divider::before, .rs-divider::after {content:""; flex:1; height:1px; background:#E2E8F0;}
.rs-feats {display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin-top:26px;}
.rs-feat {display:flex; gap:12px; align-items:flex-start; background:rgba(255,255,255,.7); border:1px solid #E2E8F0; border-radius:16px; padding:14px;}
.rs-feat .ic {flex:none; width:38px; height:38px; border-radius:11px; display:grid; place-items:center; background:#CCFBF1; color:#0F766E;}
.rs-feat b {display:block; color:#0F172A; font-size:.95rem;}
.rs-feat span {color:#64748B; font-size:.84rem;}

/* clickable sample photos: the (invisible) button covers the whole thumbnail */
.st-key-samples [data-testid="stHorizontalBlock"], .st-key-samples_more [data-testid="stHorizontalBlock"] {flex-wrap:nowrap; gap:10px;}
.st-key-samples [data-testid="stColumn"], .st-key-samples_more [data-testid="stColumn"] {position:relative; min-width:0;}
.st-key-samples [class*="st-key-pick_"], .st-key-samples_more [class*="st-key-pick_"] {position:absolute; inset:0; z-index:3;}
.st-key-samples [class*="st-key-pick_"] .stButton, .st-key-samples [class*="st-key-pick_"] button,
.st-key-samples_more [class*="st-key-pick_"] .stButton, .st-key-samples_more [class*="st-key-pick_"] button {width:100%; height:100%; opacity:0; cursor:pointer;}
.st-key-samples img, .st-key-samples_more img {border-radius:16px; transition:transform .2s ease, box-shadow .2s ease; box-shadow:0 2px 6px rgba(15,23,42,.10);}
.st-key-samples [data-testid="stColumn"]:hover img, .st-key-samples_more [data-testid="stColumn"]:hover img {transform:translateY(-4px); box-shadow:0 12px 26px rgba(15,23,42,.20);}

/* result */
.rs-card {background:#fff; border:1px solid #E2E8F0; border-radius:22px; padding:22px; box-shadow:0 1px 2px rgba(15,23,42,.04), 0 10px 30px rgba(15,23,42,.06);}
.rs-res-head {display:flex; align-items:center; gap:18px;}
.rs-ring {flex:none; width:118px; text-align:center;}
.rs-ring small {display:block; color:#64748B; font-size:.8rem; font-weight:600; margin-top:2px;}
.rs-kicker {font-size:.72rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; color:#64748B;}
.rs-stage {font-size:1.55rem; font-weight:800; letter-spacing:-.02em; color:#0F172A; line-height:1.2; margin:3px 0 10px;}
.rs-pill {display:inline-flex; align-items:center; gap:6px; padding:5px 12px; border-radius:999px; font-size:.84rem; font-weight:700;}
.rs-next {display:flex; gap:12px; margin-top:18px; padding:14px 16px; border-radius:16px; background:var(--gbg);}
.rs-next .ic {flex:none; width:34px; height:34px; border-radius:10px; display:grid; place-items:center; background:var(--g); color:#fff;}
.rs-next b {display:block; color:#0F172A; margin-bottom:2px;}
.rs-next p {margin:0; color:#334155; font-size:.95rem; line-height:1.5;}
.rs-warn {display:flex; gap:10px; align-items:center; margin-top:12px; background:#FFF7ED; border:1px solid #FED7AA; color:#9A3412;
  border-radius:14px; padding:10px 14px; font-size:.88rem;}
.rs-scale {display:grid; grid-template-columns:repeat(5,1fr); gap:6px; margin-top:20px; align-items:end;}
.rs-seg .bar {height:8px; border-radius:6px; opacity:.22;}
.rs-seg.on .bar {height:14px; opacity:1;}
.rs-seg .lbl {font-size:.74rem; color:#94A3B8; text-align:center; margin-top:6px;}
.rs-seg.on .lbl {color:#0F172A; font-weight:700;}

/* photo with an instant heat-map switch (pure CSS, no reload) */
.rs-viewer {position:relative;}
.rs-viewer input {position:absolute; opacity:0; pointer-events:none;}
.rs-imgs {position:relative; border-radius:22px; overflow:hidden; box-shadow:0 10px 30px rgba(15,23,42,.14); background:#000;}
.rs-imgs img {display:block; width:100%; height:auto;}
.rs-imgs .heat {position:absolute; inset:0; opacity:0; transition:opacity .35s ease;}
.rs-viewer input:checked ~ .rs-imgs .heat {opacity:1;}
.rs-switch {display:flex; align-items:center; gap:10px; margin-top:14px; cursor:pointer; font-weight:600; color:#0F172A; font-size:.95rem; user-select:none;}
.rs-switch .track {width:44px; height:24px; border-radius:999px; background:#CBD5E1; position:relative; transition:background .2s; flex:none;}
.rs-switch .track::after {content:""; position:absolute; top:3px; left:3px; width:18px; height:18px; border-radius:50%; background:#fff;
  box-shadow:0 1px 3px rgba(0,0,0,.25); transition:transform .2s;}
.rs-viewer input:checked ~ .rs-switch .track {background:#0F766E;}
.rs-viewer input:checked ~ .rs-switch .track::after {transform:translateX(20px);}
.rs-heat-note {display:none; color:#64748B; font-size:.84rem; margin-top:6px;}
.rs-viewer input:checked ~ .rs-heat-note {display:block;}

/* tiles, probability bars, legends */
.rs-tiles {display:grid; gap:10px; margin:8px 0 14px;}
.rs-tile {background:#fff; border:1px solid #E2E8F0; border-radius:16px; padding:14px 16px; box-shadow:0 1px 2px rgba(15,23,42,.04);}
.rs-tile .v {font-size:1.45rem; font-weight:800; color:#0F172A; line-height:1.2; letter-spacing:-.02em;}
.rs-tile .l {font-size:.8rem; color:#64748B; margin-top:2px;}
.rs-prob {display:grid; grid-template-columns:130px 1fr 56px; align-items:center; gap:12px; margin:10px 0; font-size:.92rem; color:#475569;}
.rs-prob b {text-align:right; font-weight:600; color:#334155;}
.rs-prob.on, .rs-prob.on b {color:#0F172A; font-weight:800;}
.rs-track {height:10px; background:#EEF2F6; border-radius:999px; overflow:hidden;}
.rs-track div {height:100%; border-radius:999px;}
.rs-dist {display:flex; height:16px; border-radius:999px; overflow:hidden; margin:4px 0 8px; background:#EEF2F6;}
.rs-legend span {display:inline-flex; align-items:center; gap:6px; margin:0 14px 4px 0; font-size:.82rem; color:#475569;}
.rs-legend i {width:10px; height:10px; border-radius:3px; display:inline-block;}
.rs-section {font-size:1.3rem; font-weight:800; letter-spacing:-.02em; color:#0F172A; margin:6px 0 4px;}

/* chat */
[data-testid="stChatMessage"] {background:#fff; border:1px solid #E2E8F0; border-radius:18px; padding:14px 16px; box-shadow:0 1px 2px rgba(15,23,42,.04);}
.rs-context {display:flex; gap:10px; align-items:center; background:#F0FDFA; border:1px solid #99F6E4; color:#115E59; border-radius:16px; padding:12px 16px; font-size:.93rem;}
.rs-context.muted {background:#fff; border-color:#E2E8F0; color:#475569;}
.rs-footer {text-align:center; color:#94A3B8; font-size:.8rem; margin-top:34px;}

/* responsive rows */
.st-key-result_row {margin-bottom:14px;}
.st-key-result_row [data-testid="stHorizontalBlock"] {flex-wrap:wrap;}
.st-key-result_row [data-testid="stColumn"] {min-width:min(320px,100%);}
.st-key-stages [data-testid="stHorizontalBlock"] {flex-wrap:nowrap;}
.st-key-stages [data-testid="stColumn"] {min-width:0;}
@media (max-width: 720px) {.rs-hero .t {font-size:1.75rem;} .rs-res-head {flex-direction:column; align-items:flex-start;}
  .rs-prob {grid-template-columns:100px 1fr 50px;} .rs-proto {display:none;}}
</style>
"""


def html(markup: str) -> None:
    """Render app-generated HTML. Markup must not contain blank lines (Markdown would end the block)."""
    st.markdown(markup, unsafe_allow_html=True)


def data_uri(rgb: np.ndarray, size: int = 448) -> str:
    """Encode an image as an inline JPEG for the HTML parts of the page."""
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def tiles(items, cols: int = 2) -> str:
    cells = "".join(f'<div class="rs-tile"><div class="v">{v}</div><div class="l">{label}</div></div>'
                    for label, v in items)
    # two tiles per row are kept fixed; wider rows wrap to fewer columns when space is short
    grid = "repeat(2,minmax(0,1fr))" if cols == 2 else f"repeat(auto-fit,minmax({110 if cols == 3 else 150}px,1fr))"
    return f'<div class="rs-tiles" style="grid-template-columns:{grid}">{cells}</div>'


def hero(title: str, accent: str, text: str) -> None:
    html(f'<div class="rs-hero"><div class="t">{title} <span>{accent}</span></div><p>{text}</p></div>')


def ring(grade: int, conf: float, colour: str) -> str:
    """Confidence ring with the grade in the centre."""
    circ = 2 * np.pi * 50
    return (f'<div class="rs-ring"><svg width="118" height="118" viewBox="0 0 120 120">'
            f'<circle cx="60" cy="60" r="50" fill="none" stroke="#EEF2F6" stroke-width="10"/>'
            f'<circle cx="60" cy="60" r="50" fill="none" stroke="{colour}" stroke-width="10" stroke-linecap="round" '
            f'stroke-dasharray="{circ * conf:.1f} {circ:.1f}" transform="rotate(-90 60 60)"/>'
            f'<text x="60" y="50" text-anchor="middle" font-size="11" font-weight="700" fill="#64748B" letter-spacing="1.5">GRADE</text>'
            f'<text x="60" y="84" text-anchor="middle" font-size="40" font-weight="800" fill="{colour}">{grade}</text>'
            f'</svg><small>{conf:.0%} confidence</small></div>')


def severity_scale(active: int) -> str:
    cells = "".join(
        f'<div class="rs-seg{" on" if i == active else ""}"><div class="bar" style="background:{GRADE_COLOURS[i]}"></div>'
        f'<div class="lbl">{FRIENDLY[i]}</div></div>' for i in range(5))
    return f'<div class="rs-scale">{cells}</div>'


def result_card(res: dict) -> str:
    """Headline result for a non-specialist: grade, stage, urgency, what to do next, place on the scale."""
    g, conf = res["grade"], res["confidence"]
    s, col = STAGE_INFO[g], GRADE_COLOURS[g]
    parts = [f'<div class="rs-card" style="--g:{col};--gbg:{col}14">',
             '<div class="rs-res-head">', ring(g, conf, col),
             '<div><div class="rs-kicker">Result</div>'
             f'<div class="rs-stage">{s["name"]}</div>'
             f'<span class="rs-pill" style="color:{col};background:{col}1F">● {s["urgency"]}</span></div></div>',
             f'<div class="rs-next"><div class="ic">{icon("next", 18)}</div><div><b>What to do next</b>'
             f'<p>{s["advice"].capitalize()}</p></div></div>']
    if conf < LOW_CONFIDENCE:
        parts.append(f'<div class="rs-warn">{icon("alert", 18)}<span>The AI is unsure about this photo. '
                     'Please ask an eye-care professional to look at it.</span></div>')
    parts.append(severity_scale(g))
    parts.append("</div>")
    return "".join(parts)


def photo_viewer(res: dict) -> str:
    """The photograph with a switch that fades the Grad-CAM heat-map in and out instantly (CSS only)."""
    base, heat = data_uri(res["processed"]), data_uri(overlay_heatmap(res["processed"], res["heatmap"], 0.45))
    return ('<div class="rs-viewer"><input type="checkbox" id="rs-heat">'
            f'<div class="rs-imgs"><img src="{base}" alt="Your photograph"><img class="heat" src="{heat}" alt="Heat-map"></div>'
            '<label class="rs-switch" for="rs-heat"><span class="track"></span>Show where the AI looked</label>'
            '<div class="rs-heat-note">Red areas influenced the result most, blue areas least (Grad-CAM). '
            'In diseased eyes they usually cover lesions.</div></div>')


def prob_bars(probs, active: int) -> str:
    return "".join(
        f'<div class="rs-prob{" on" if i == active else ""}"><span>{i} · {FRIENDLY[i]}</span>'
        f'<div class="rs-track"><div style="width:{max(float(p) * 100, 0.8):.1f}%;background:{GRADE_COLOURS[i]}"></div></div>'
        f'<b>{p:.1%}</b></div>' for i, p in enumerate(probs))


def grade_legend() -> str:
    return '<div class="rs-legend">' + "".join(
        f'<span><i style="background:{GRADE_COLOURS[g]}"></i>{g} · {FRIENDLY[g]}</span>' for g in range(5)) + "</div>"


def sample_grid(paths, key: str) -> None:
    """Row(s) of sample thumbnails; clicking a thumbnail grades it."""
    with st.container(key=key):
        for i in range(0, len(paths), 5):
            cols = st.columns(5)
            for col, p in zip(cols, paths[i:i + 5]):
                ref = sample_info(p.name)[1]
                with col:
                    st.image(thumbnail(str(p), ref), width="stretch")
                    st.button(f"Try sample: grade {ref}", key=f"pick_{key}_{p.name}", on_click=choose_sample, args=(p.name,))


# --------------------------------------------------------------------------- #
# Page frame
# --------------------------------------------------------------------------- #
html(CSS)
model, meta = load_model_and_meta()
ss = st.session_state
samples = (sorted(SAMPLES_DIR.glob("*.png")) + sorted(SAMPLES_DIR.glob("*.jpg"))) if SAMPLES_DIR.exists() else []

html(f'<div class="rs-top"><div class="rs-brand"><div class="rs-mark">{icon("eye", 22, "#fff")}</div>'
     '<div><b>RetinaScreen</b><small>AI screening for diabetic retinopathy</small></div></div>'
     '<span class="rs-proto">Coursework prototype</span></div>')

if model is None:
    st.error("No trained model found in `models/`. Run `scripts/train.py --save-best` first.")
    st.stop()

tab_single, tab_batch, tab_bot, tab_card = st.tabs([":material/center_focus_strong: Screen a photo",
                                                    ":material/grid_view: Batch screening",
                                                    ":material/forum: RetinaBot", ":material/info: About"])

# ---------------------------------------------------------------------- #
# Tab 1 - single image: choose a photo, then see the result
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
        with st.container(key="landing"):
            hero("Check a retina photo for", "diabetic retinopathy",
                 "Upload a photograph of the back of the eye. In a few seconds you see the stage of the disease "
                 "and what to do next.")
            st.file_uploader("Retina photograph", type=["png", "jpg", "jpeg"], label_visibility="collapsed",
                             key=f"upload_{ss['upload_key']}", on_change=store_upload)
            if samples:
                featured = []
                for g in range(5):        # one APTOS test photo of each stage
                    featured += [p for p in samples if p.name.lower().startswith(f"aptos_grade{g}_")][:1]
                others = [p for p in samples if p not in featured]
                html('<div class="rs-divider">No photo to hand? Tap a sample, one of each stage</div>')
                sample_grid(featured, "samples")
                if others:
                    with st.expander("More samples, including photos from another country"):
                        sample_grid(others, "samples_more")
            html('<div class="rs-feats">'
                 f'<div class="rs-feat"><div class="ic">{icon("upload")}</div><div><b>Add a photo</b>'
                 '<span>A colour photograph of the retina.</span></div></div>'
                 f'<div class="rs-feat"><div class="ic">{icon("pulse")}</div><div><b>AI grades it</b>'
                 '<span>On the 5-stage international scale.</span></div></div>'
                 f'<div class="rs-feat"><div class="ic">{icon("check")}</div><div><b>Know what to do</b>'
                 '<span>Whether, and how soon, to see a specialist.</span></div></div></div>')
    else:
        name, data = ss["photo"]
        with st.spinner("Checking the photograph ..."):
            res = grade_bytes(data, meta.get("run_name", ""), model, meta)
        ss["last_result"] = res

        st.button("Check another photo", icon=":material/arrow_back:", type="tertiary", on_click=clear_photo)
        with st.container(key="result_row"):
            photo_col, res_col = st.columns([5, 6], gap="large")
            with photo_col:
                html(photo_viewer(res))
            with res_col:
                html(result_card(res))
                with st.container(horizontal=True, vertical_alignment="center"):
                    st.download_button("Download report", screening_summary(name, res, meta), "screening_summary.txt",
                                       "text/plain", type="primary", icon=":material/download:")
                    st.caption("Questions? Ask **RetinaBot** in the menu above.")

        with st.expander("More details"):
            t_prob, t_tech = st.tabs(["How sure is the AI?", "Technical details"])
            with t_prob:
                st.caption("Probability the AI gave to each stage.")
                html(prob_bars(res["probs"], res["grade"]))
            with t_tech:
                src_label, ref = sample_info(name)
                if ref is not None:
                    agree = "agrees" if ref == res["grade"] else ("is one grade apart" if abs(ref - res["grade"]) == 1
                                                                  else "disagrees")
                    st.markdown(f"**Reference grade** in the {src_label}: {ref} - {C.CLASS_LABELS[ref]} "
                                f"(the prediction {agree}).")
                st.markdown(f"**File:** {escape(name)}  \n**Model:** {meta['backbone']}, {meta['preprocess']} "
                            f"preprocessing, {meta['img_size']} px input"
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
    with st.container(key="batch_top"):
        hero("Screen a whole clinic", "at once",
             "Add many photographs. The most urgent eyes are listed first, and you can download the results.")
        files = st.file_uploader("Retina photographs", type=["png", "jpg", "jpeg"], accept_multiple_files=True,
                                 label_visibility="collapsed", key=f"batch_{ss['batch_key']}")
        if files:
            ss["batch_samples"] = False
        with st.container(horizontal=True, horizontal_alignment="center"):
            if samples:
                st.button(f"Try it with our {len(samples)} sample photos", icon=":material/photo_library:",
                          on_click=use_batch_samples)
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
        html(tiles([("photos checked", n),
                    ("need referral", f"{n_ref} <span style='font-size:.9rem;color:#64748B'>({n_ref / n:.0%})</span>"),
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
        st.download_button("Download results (CSV)", df.to_csv(index=False).encode(), "dr_batch_report.csv",
                           "text/csv", type="primary", icon=":material/download:")
        with st.expander("More details"):
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
            cols = st.columns(min(5, len(thumbs)))
            for i, (name, img, g, conf) in enumerate(thumbs):
                cols[i % len(cols)].image(img, caption=f"Grade {g} ({conf:.0%})", width="stretch")

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
            html(f'<div class="rs-context">{icon("check", 18)}<span>Talking about your result: <b>Grade {last["grade"]} · '
                 f'{STAGE_INFO[last["grade"]]["name"]}</b></span></div>')
        else:
            html('<div class="rs-context muted">Check a photo first to ask about your result, '
                 'or ask anything about diabetic retinopathy.</div>')
    if h2.button("Clear chat", type="tertiary", icon=":material/refresh:"):
        ss["chat"] = [("assistant", bot.greeting(ChatContext()))]
        st.rerun()
    for role, text in ss["chat"]:
        with st.chat_message(role, avatar=":material/visibility:" if role == "assistant" else ":material/person:"):
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
    html('<div class="rs-section">How accurate is it?</div>')
    c1, c2 = st.columns(2, gap="large")
    for col, title, mm in [(c1, "New photos from the same hospital (APTOS 2019)", m),
                           (c2, "Photos from another country (EyePACS 2015)", ext)]:
        if mm:
            col.markdown(f"**{title}**")
            with col:
                html(tiles([(label, fmt.format(metric(mm, key))) for label, key, fmt in keys], cols=2))
    st.caption("Accuracy drops on photographs from other cameras and populations, so a clinic would need to check "
               "and adapt the model with its own photographs before use. RetinaScreen is a coursework prototype, "
               "not a medical device.")

    with st.expander("How the model was built", icon=":material/neurology:"):
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
    with st.expander("Intended use, limitations and ethics", icon=":material/balance:"):
        st.markdown("""
**Intended use** - decision *support* for screening programmes: prioritise which patients an
ophthalmologist should see first.  Not a diagnostic device.

**Limitations** - developed on one source (Aravind Eye Hospital, India) and validated externally
on EyePACS (USA); cannot detect diabetic macular oedema or non-DR pathology; adjacent grades are
frequently confused; image quality strongly affects results; model confidence is not clinical certainty.

**Ethics** - both datasets are anonymised and released under CC0; predictions are explained with
Grad-CAM so a clinician can verify that the evidence is anatomically plausible.
""")
    with st.expander("All metrics", icon=":material/table_chart:"):
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
        with st.expander("Training and evaluation evidence (figures)", icon=":material/insert_chart:"):
            for tab, (_, figs) in zip(st.tabs([t for t, _ in evidence]), evidence):
                with tab:
                    for p, cap in figs:
                        st.image(str(p), caption=cap, width="stretch")

html('<div class="rs-footer">RetinaScreen · university coursework prototype · not a medical device</div>')

"""
Grade one or more fundus images with the trained model (no Streamlit needed).

Usage
-----
    python scripts/predict.py path/to/image.png
    python scripts/predict.py folder_with_images/ --csv predictions.csv --gradcam out_dir/

Uses models/best_model.keras + models/model_metadata.json produced by
``scripts/train.py --save-best`` (or the Colab notebook).
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from dr_detection import config as C
from dr_detection.gradcam import make_gradcam_heatmap, overlay_heatmap
from dr_detection.preprocessing import load_image, preprocess_image
from dr_detection.utils import load_json

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="image files or folders")
    ap.add_argument("--models-dir", default=C.Config().models_dir)
    ap.add_argument("--csv", default=None, help="write predictions to this CSV")
    ap.add_argument("--gradcam", default=None, help="folder to write Grad-CAM overlays")
    args = ap.parse_args()

    from tensorflow import keras

    meta = load_json(Path(args.models_dir) / "model_metadata.json")
    model = keras.models.load_model(Path(args.models_dir) / "best_model.keras")

    paths = []
    for inp in args.inputs:
        p = Path(inp)
        if p.is_dir():
            paths += sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_EXTENSIONS)
        else:
            paths.append(p)

    rows = []
    for p in paths:
        img = preprocess_image(load_image(p), meta["preprocess"], meta["img_size"])
        heatmap, probs, k = make_gradcam_heatmap(model, img)
        rows.append({"image": p.name, "grade": k, "class": C.CLASS_LABELS[k], "confidence": float(probs[k]),
                     "referable": bool(k >= C.REFERABLE_THRESHOLD),
                     **{f"p_{C.CLASS_NAMES[i]}": float(probs[i]) for i in range(5)}})
        print(f"{p.name:40s} -> grade {k} ({C.CLASS_LABELS[k]}, {probs[k]:.1%})")
        if args.gradcam:
            out = Path(args.gradcam)
            out.mkdir(parents=True, exist_ok=True)
            side_by_side = np.concatenate([img, overlay_heatmap(img, heatmap)], axis=1)
            cv2.imwrite(str(out / f"{p.stem}_gradcam.png"), cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))

    if args.csv:
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f"[predict] wrote {args.csv}")


if __name__ == "__main__":
    main()

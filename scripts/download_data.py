"""
Download and unpack the Kaggle dataset.

Usage
-----
    python scripts/download_data.py            # -> data/diabetic_retinopathy_data
    python scripts/download_data.py --out /content/data   (Colab)

Requires a Kaggle API token in ``~/.kaggle/kaggle.json`` (Kaggle -> Settings ->
API -> Create New Token).  The token is only ever read by the official Kaggle
CLI; this script never prints or copies it.
"""

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import _bootstrap  # noqa: F401
from dr_detection import config as C


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path(C.Config().data_dir)), help="destination folder")
    ap.add_argument("--dataset", default=C.KAGGLE_DATASET)
    ap.add_argument("--keep-zip", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if (out / "colored_images").exists():
        print(f"[download] dataset already present at {out}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)

    if shutil.which("kaggle") is None:
        sys.exit("kaggle CLI not found - run:  pip install kaggle")

    print(f"[download] kaggle datasets download -d {args.dataset}")
    subprocess.run(["kaggle", "datasets", "download", "-d", args.dataset, "-p", str(out.parent)], check=True)

    zip_path = out.parent / C.KAGGLE_ZIP_NAME
    print(f"[download] extracting {zip_path} -> {out}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out)
    if not args.keep_zip:
        zip_path.unlink()

    classes = sorted(p.name for p in (out / "colored_images").iterdir() if p.is_dir())
    counts = {c: len(list((out / "colored_images" / c).glob("*.png"))) for c in classes}
    print("[download] done. Images per folder:", counts, " total:", sum(counts.values()))


if __name__ == "__main__":
    main()

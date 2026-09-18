"""
Bundle everything the report and the demo app need into one zip file.

Included: results/ (metrics, figures, histories - but not the per-run model
checkpoints, which are large) and models/best_model.keras + model_metadata.json.

Usage:  python scripts/export_results.py [--out dr_results_bundle.zip] [--with-run-models]

On Colab the produced zip can be downloaded with
    from google.colab import files; files.download("dr_results_bundle.zip")
"""

import argparse
import zipfile
from pathlib import Path

import _bootstrap  # noqa: F401
from dr_detection import config as C


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="dr_results_bundle.zip")
    ap.add_argument("--with-run-models", action="store_true", help="also include results/*/model.keras")
    args = ap.parse_args()

    cfg = C.Config()
    root = Path(cfg.results_dir).parent
    n = 0
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in Path(cfg.results_dir).rglob("*"):
            if p.is_dir():
                continue
            if p.suffix == ".keras" and not args.with_run_models:
                continue
            if p.name.startswith("checkpoint_"):
                continue
            zf.write(p, p.relative_to(root))
            n += 1
        for name in ("best_model.keras", "model_metadata.json"):
            p = Path(cfg.models_dir) / name
            if p.exists():
                zf.write(p, p.relative_to(root))
                n += 1
    print(f"[export] {n} files -> {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

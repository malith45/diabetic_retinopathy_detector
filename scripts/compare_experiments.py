"""
Run the experiment grid used in the report and build the comparison table.

Two questions are answered:

1. **Which preprocessing helps most?**   EfficientNetB0 trained on
   raw / CLAHE / Ben Graham / CLAHE+Ben Graham images.
2. **Which backbone is best?**           Ben Graham images with
   EfficientNetB0 / EfficientNetB3 / ResNet50V2 / MobileNetV2 / DenseNet121.

All runs share the same seed and therefore the *same* train / val / test
images, so the numbers are directly comparable.  Preprocessed images are cached
per method so each preprocessing is computed only once.

Outputs
-------
    results/experiments_comparison.csv
    results/experiments_comparison.png
    results/<run_name>/...   (one folder per run, see scripts/train.py)

Usage
-----
    python scripts/compare_experiments.py                       # full grid
    python scripts/compare_experiments.py --quick-test          # smoke test
    python scripts/compare_experiments.py --only preprocessing  # or backbones
"""

import argparse
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from dr_detection import config as C
from dr_detection.data import prepare_data
from dr_detection.train import run_experiment
from dr_detection.visualize import plot_experiment_comparison

PREPROCESS_GRID = ["raw", "clahe", "ben_graham", "clahe_ben_graham"]
BACKBONE_GRID = ["EfficientNetB0", "EfficientNetB3", "ResNet50V2", "MobileNetV2", "DenseNet121"]

SUMMARY_COLUMNS = [
    "run_name", "backbone", "preprocess", "accuracy", "macro_precision", "macro_recall", "macro_f1",
    "weighted_f1", "quadratic_weighted_kappa", "referable_sensitivity", "referable_specificity",
    "within_one_grade_accuracy", "epochs_trained", "train_minutes",
]


def collect_results(results_dir: Path) -> pd.DataFrame:
    """Read every results/<run>/metrics.json into one table."""
    import json

    rows = []
    for mfile in sorted(results_dir.glob("*/metrics.json")):
        with open(mfile, encoding="utf-8") as f:
            m = json.load(f)
        rows.append({k: m.get(k) for k in SUMMARY_COLUMNS})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=C.Config().data_dir)
    ap.add_argument("--results-dir", default=C.Config().results_dir)
    ap.add_argument("--only", choices=["preprocessing", "backbones", "collect"], default=None)
    ap.add_argument("--epochs-phase1", type=int, default=C.Config().epochs_phase1)
    ap.add_argument("--epochs-phase2", type=int, default=C.Config().epochs_phase2)
    ap.add_argument("--quick-test", action="store_true")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    common = dict(data_dir=args.data_dir, results_dir=args.results_dir, quick_test=args.quick_test,
                  epochs_phase1=args.epochs_phase1, epochs_phase2=args.epochs_phase2)

    if args.only != "collect":
        grid = []
        if args.only in (None, "preprocessing"):
            grid += [("EfficientNetB0", p) for p in PREPROCESS_GRID]
        if args.only in (None, "backbones"):
            grid += [(b, "ben_graham") for b in BACKBONE_GRID]
        grid = list(dict.fromkeys(grid))  # de-duplicate, keep order

        data_cache = {}
        for backbone, prep in grid:
            cfg = C.Config(backbone=backbone, preprocess=prep, **common)
            if (cfg.run_dir / "metrics.json").exists():
                print(f"[compare] skipping {cfg.run_name} (already done)")
                continue
            if prep not in data_cache:
                data_cache[prep] = prepare_data(cfg)
            run_experiment(cfg, data=data_cache[prep], make_figures=True)

    table = collect_results(results_dir)
    table = table[~table["run_name"].str.contains("quicktest")] if not args.quick_test else table
    table = table.sort_values("quadratic_weighted_kappa", ascending=False)
    table.to_csv(results_dir / "experiments_comparison.csv", index=False)
    plot_experiment_comparison(table, results_dir / "experiments_comparison.png")
    pd.set_option("display.width", 200)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

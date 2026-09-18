"""
Produce every exploratory-data-analysis and pipeline figure used in the report.

These figures do **not** need a trained model or a GPU:

    results/figures/
        class_distribution.png        - imbalance of the 5 grades
        split_distribution.png        - stratified 70/10/20 split per class
        sample_grid_raw.png           - 4 raw images per grade
        sample_grid_ben_graham.png    - the same after preprocessing
        preprocessing_stages.png      - one image through every stage
        preprocessing_comparison.png  - raw vs CLAHE vs Ben Graham per grade
        intensity_stats.png           - brightness / contrast variability
        augmentation_grid.png         - random augmentations of one image
        class_balance.png             - training counts + class weights
        dataset_summary.json          - numbers used in the report tables

Usage:  python scripts/make_figures.py [--data-dir ...] [--out results/figures]
"""

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from dr_detection import config as C
from dr_detection import visualize as V
from dr_detection.augmentation import compute_class_weights
from dr_detection.data import build_dataframe, class_distribution, cross_check_with_csv, split_dataframe
from dr_detection.utils import ensure_dir, save_json, set_seed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default=str(Path(C.Config().results_dir) / "figures"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = C.Config(**({"data_dir": args.data_dir} if args.data_dir else {}))
    out = ensure_dir(args.out)

    df = build_dataframe(cfg.images_dir)
    df = cross_check_with_csv(df, cfg.data_dir)
    train, val, test = split_dataframe(df, cfg.val_frac, cfg.test_frac, cfg.seed)
    print(class_distribution(df).to_string(index=False))

    V.plot_class_distribution(df, out / "class_distribution.png")
    V.plot_split_distribution(train, val, test, out / "split_distribution.png")
    V.plot_sample_grid(df, out / "sample_grid_raw.png", method="raw", seed=args.seed)
    V.plot_sample_grid(df, out / "sample_grid_ben_graham.png", method="ben_graham", seed=args.seed)
    example = df[df["label"] == 2].sample(1, random_state=args.seed)["image_path"].iloc[0]
    V.plot_preprocessing_stages(example, out / "preprocessing_stages.png")
    V.plot_preprocessing_comparison(df, out / "preprocessing_comparison.png", seed=args.seed)
    stats = V.plot_intensity_stats(df, out / "intensity_stats.png", seed=args.seed)
    V.plot_augmentation_grid(example, out / "augmentation_grid.png")
    weights = compute_class_weights(train["label"].to_numpy())
    V.plot_class_balance(train["label"].to_numpy(), weights, out / "class_balance.png")

    summary = {
        "kaggle_dataset": C.KAGGLE_DATASET,
        "total_images": int(len(df)),
        "class_distribution": class_distribution(df).to_dict(orient="records"),
        "split_sizes": {"train": int(len(train)), "val": int(len(val)), "test": int(len(test))},
        "split_per_class": {
            name: {C.CLASS_LABELS[c]: int((part["label"] == c).sum()) for c in range(5)}
            for name, part in (("train", train), ("val", val), ("test", test))
        },
        "class_weights": weights,
        "image_size_px": {"height": int(stats["height"].mode()[0]), "width": int(stats["width"].mode()[0])},
        "intensity_stats": stats.groupby("class")[["mean_brightness", "contrast_std"]].mean().round(1).to_dict(),
        "example_image": example,
    }
    save_json(summary, out / "dataset_summary.json")
    print(f"[figures] written to {out}")


if __name__ == "__main__":
    main()

"""
Train and evaluate one model from the command line.

Examples
--------
    # main experiment (EfficientNetB0 + Ben Graham, two-phase training)
    python scripts/train.py --save-best

    # a different backbone / preprocessing
    python scripts/train.py --backbone ResNet50V2 --preprocess clahe

    # 90-second smoke test on a CPU laptop
    python scripts/train.py --quick-test

Every run writes to  results/<backbone>_<preprocess>_<img_size>/ :
    config.json, history.csv, training_curves.png, metrics.json,
    classification_report.txt, confusion_matrix*.png, per_class_metrics.png,
    roc_curves.png, misclassified.png, gradcam_test_samples.png, model.keras
"""

import argparse

import _bootstrap  # noqa: F401
from dr_detection import config as C
from dr_detection.train import run_experiment


def parse_args() -> argparse.Namespace:
    d = C.Config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=d.data_dir)
    ap.add_argument("--results-dir", default=d.results_dir)
    ap.add_argument("--models-dir", default=d.models_dir)
    ap.add_argument("--backbone", default=d.backbone, choices=C.BACKBONES)
    ap.add_argument("--preprocess", default=d.preprocess, choices=C.PREPROCESS_METHODS)
    ap.add_argument("--img-size", type=int, default=d.img_size)
    ap.add_argument("--batch-size", type=int, default=d.batch_size)
    ap.add_argument("--epochs-phase1", type=int, default=d.epochs_phase1)
    ap.add_argument("--epochs-phase2", type=int, default=d.epochs_phase2)
    ap.add_argument("--lr-phase1", type=float, default=d.lr_phase1)
    ap.add_argument("--lr-phase2", type=float, default=d.lr_phase2)
    ap.add_argument("--unfreeze-layers", type=int, default=d.unfreeze_layers)
    ap.add_argument("--dropout", type=float, default=d.dropout)
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--oversample", action="store_true", help="random over-sampling instead of / in addition to class weights")
    ap.add_argument("--no-augmentation", action="store_true")
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--run-name", default="")
    ap.add_argument("--quick-test", action="store_true")
    ap.add_argument("--save-best", action="store_true", help="copy the model to models/best_model.keras for the app")
    return ap.parse_args()


def main() -> None:
    a = parse_args()
    cfg = C.Config(
        data_dir=a.data_dir, results_dir=a.results_dir, models_dir=a.models_dir,
        backbone=a.backbone, preprocess=a.preprocess, img_size=a.img_size, batch_size=a.batch_size,
        epochs_phase1=a.epochs_phase1, epochs_phase2=a.epochs_phase2,
        lr_phase1=a.lr_phase1, lr_phase2=a.lr_phase2, unfreeze_layers=a.unfreeze_layers,
        dropout=a.dropout, use_class_weights=not a.no_class_weights, oversample_minority=a.oversample,
        use_augmentation=not a.no_augmentation, seed=a.seed, run_name=a.run_name, quick_test=a.quick_test,
    )
    run_experiment(cfg, save_as_best=a.save_best)


if __name__ == "__main__":
    main()

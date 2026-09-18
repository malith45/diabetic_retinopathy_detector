"""
Two-phase transfer-learning training strategy.

Phase 1 - *feature extraction*
    The ImageNet backbone is frozen; only the new classification head is
    trained (Adam, lr = 1e-4).  This lets the randomly initialised head settle
    without destroying the pretrained filters.

Phase 2 - *fine-tuning*
    The top ``unfreeze_layers`` layers of the backbone are un-frozen and the
    whole network is trained with a 10x smaller learning rate (1e-5), so the
    generic ImageNet features are gently adapted to retinal texture.

Over-fitting control / experimental design
    * validation split monitored every epoch
    * ``EarlyStopping`` on validation loss (restores the best weights)
    * ``ReduceLROnPlateau`` halves the learning rate when validation loss stalls
    * ``ModelCheckpoint`` keeps the best model on disk
    * ``CSVLogger`` writes a per-epoch log used to plot the curves
    * dropout in the head, on-the-fly augmentation and class weights
    * the test split is evaluated exactly once, after training
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import config as C
from .augmentation import build_augmentation, compute_class_weights, oversample_minority, balance_summary
from .data import make_tf_dataset, prepare_data
from .model import build_model, compile_model, count_params, set_fine_tuning
from .utils import ensure_dir, hardware_info, save_json, set_seed, timer


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #
def get_callbacks(cfg: C.Config, run_dir: Path, phase: int):
    """Standard callback set for one training phase."""
    from tensorflow import keras

    ckpt = run_dir / f"checkpoint_phase{phase}.keras"
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=cfg.reduce_lr_patience,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
        keras.callbacks.CSVLogger(str(run_dir / f"log_phase{phase}.csv"), append=False),
    ]


def _history_to_frame(history, phase: int, epoch_offset: int) -> pd.DataFrame:
    df = pd.DataFrame(history.history)
    df.insert(0, "epoch", np.arange(1, len(df) + 1) + epoch_offset)
    df.insert(1, "phase", phase)
    return df


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_two_phase(model, base, train_ds, val_ds, cfg: C.Config, run_dir: Path,
                    class_weights: Optional[Dict[int, float]] = None) -> pd.DataFrame:
    """Run phase 1 then phase 2 and return the concatenated epoch history."""
    epochs1 = 1 if cfg.quick_test else cfg.epochs_phase1
    epochs2 = 1 if cfg.quick_test else cfg.epochs_phase2

    # ---- phase 1 ------------------------------------------------------------
    print(f"\n===== Phase 1: frozen backbone, lr={cfg.lr_phase1}, epochs={epochs1} =====")
    compile_model(model, cfg.lr_phase1)
    print("params:", count_params(model))
    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs1,
        class_weight=class_weights,
        callbacks=get_callbacks(cfg, run_dir, phase=1),
        shuffle=False,  # the tf.data pipeline already shuffles
        verbose=1,
    )
    hist = _history_to_frame(h1, 1, 0)

    # ---- phase 2 ------------------------------------------------------------
    n_train, n_frozen = set_fine_tuning(base, cfg.unfreeze_layers)
    print(f"\n===== Phase 2: fine-tune top {n_train} layers ({n_frozen} frozen), "
          f"lr={cfg.lr_phase2}, epochs={epochs2} =====")
    compile_model(model, cfg.lr_phase2)  # re-compile so trainable flags take effect
    print("params:", count_params(model))
    h2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs2,
        class_weight=class_weights,
        callbacks=get_callbacks(cfg, run_dir, phase=2),
        shuffle=False,
        verbose=1,
    )
    hist = pd.concat([hist, _history_to_frame(h2, 2, len(hist))], ignore_index=True)
    hist.to_csv(run_dir / "history.csv", index=False)
    return hist


def run_experiment(cfg: C.Config, data: Optional[dict] = None, save_as_best: bool = False,
                   make_figures: bool = True) -> dict:
    """End-to-end experiment: data -> model -> two-phase training -> evaluation.

    Parameters
    ----------
    cfg          : the run configuration
    data         : optional pre-loaded output of ``data.prepare_data`` (re-used
                   across runs with the same preprocessing to save time)
    save_as_best : also copy the trained model + metadata to ``cfg.models_dir``
                   as ``best_model.keras`` / ``model_metadata.json`` (used by the app)
    make_figures : produce evaluation and Grad-CAM figures

    Returns the metrics dictionary that was written to ``run_dir/metrics.json``.
    """
    from .evaluate import evaluate_model, plot_training_curves
    from .gradcam import gradcam_grid

    set_seed(cfg.seed)
    run_dir = ensure_dir(cfg.run_dir)
    save_json(cfg.to_dict(), run_dir / "config.json")
    print(f"[run] {cfg.run_name}  ->  {run_dir}")

    # ---- data -----------------------------------------------------------
    if data is None:
        data = prepare_data(cfg)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    y_train_before = y_train.copy()
    if cfg.oversample_minority:
        X_train, y_train = oversample_minority(X_train, y_train, cfg.seed)
    class_weights = compute_class_weights(y_train) if cfg.use_class_weights else None
    save_json(balance_summary(y_train_before, y_train if cfg.oversample_minority else None,
                              class_weights or {}), run_dir / "class_balance.json")
    print("[run] class weights:", class_weights)

    aug = build_augmentation(cfg.seed) if cfg.use_augmentation else None
    train_ds = make_tf_dataset(X_train, y_train, cfg.batch_size, shuffle=True, augmentation=aug, seed=cfg.seed)
    val_ds = make_tf_dataset(X_val, y_val, cfg.batch_size)

    # ---- model ----------------------------------------------------------
    model, base = build_model(cfg)
    with open(run_dir / "model_summary.txt", "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda s, **kw: f.write(s + "\n"))

    # ---- train ----------------------------------------------------------
    t0 = time.perf_counter()
    with timer("training"):
        hist = train_two_phase(model, base, train_ds, val_ds, cfg, run_dir, class_weights)
    train_minutes = (time.perf_counter() - t0) / 60
    if make_figures:
        plot_training_curves(hist, run_dir / "training_curves.png", title=cfg.run_name)

    # ---- evaluate -------------------------------------------------------
    metrics = evaluate_model(model, X_test, y_test, run_dir, batch_size=cfg.batch_size,
                             make_figures=make_figures)
    metrics.update({
        "run_name": cfg.run_name,
        "backbone": cfg.backbone,
        "preprocess": cfg.preprocess,
        "img_size": cfg.img_size,
        "epochs_trained": int(len(hist)),
        "best_val_loss": float(hist["val_loss"].min()),
        "best_val_accuracy": float(hist["val_accuracy"].max()),
        "train_minutes": round(train_minutes, 2),
        "params": count_params(model),
        "hardware": hardware_info(),
    })
    save_json(metrics, run_dir / "metrics.json")

    # ---- save model + Grad-CAM -------------------------------------------
    model_path = run_dir / "model.keras"
    model.save(model_path)
    if make_figures:
        try:
            gradcam_grid(model, X_test, y_test, run_dir / "gradcam_test_samples.png", n_per_class=2)
        except Exception as exc:  # Grad-CAM must never break a run
            print("[run] Grad-CAM failed:", exc)

    if save_as_best:
        ensure_dir(cfg.models_dir)
        model.save(Path(cfg.models_dir) / "best_model.keras")
        save_json({
            "run_name": cfg.run_name,
            "backbone": cfg.backbone,
            "preprocess": cfg.preprocess,
            "img_size": cfg.img_size,
            "class_names": C.CLASS_NAMES,
            "class_labels": C.CLASS_LABELS,
            "metrics": {k: metrics[k] for k in ("accuracy", "macro_f1", "weighted_f1",
                                                 "quadratic_weighted_kappa",
                                                 "referable_sensitivity", "referable_specificity")
                        if k in metrics},
            "trained_on": metrics["hardware"],
        }, Path(cfg.models_dir) / "model_metadata.json")
        print(f"[run] saved best model to {cfg.models_dir}")

    print(f"[run] done: acc={metrics['accuracy']:.4f}  macro-F1={metrics['macro_f1']:.4f}  "
          f"QWK={metrics['quadratic_weighted_kappa']:.4f}")
    return metrics

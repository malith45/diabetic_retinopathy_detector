"""
Model evaluation: metrics, curves, confusion matrix and error analysis.

Metrics reported
----------------
* accuracy
* precision / recall / F1 - per class, macro and weighted average
* quadratic weighted kappa (QWK) - the official APTOS 2019 metric; it rewards
  predictions that are *close* to the true grade and penalises far misses,
  which matches the ordinal nature of DR staging
* **referable-DR** sensitivity / specificity / AUC - the clinically relevant
  binary question "should this patient be referred to an ophthalmologist?"
  (grade >= 2, i.e. moderate NPDR or worse)
* **any-DR** sensitivity / specificity - grade >= 1
* one-vs-rest ROC curves and AUC per class

All figures are written as PNG at 150 dpi so they can be pasted directly into
the report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from . import config as C  # noqa: E402

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
PALETTE = ["#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#9b2226"]


# --------------------------------------------------------------------------- #
# Predictions
# --------------------------------------------------------------------------- #
def predict_probs(model, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
    """Softmax probabilities for a uint8 image array (N, H, W, 3)."""
    return model.predict(X.astype(np.float32), batch_size=batch_size, verbose=0)


def _binary_stats(y_true_bin: np.ndarray, y_pred_bin: np.ndarray, prob: Optional[np.ndarray]) -> Dict[str, float]:
    tp = int(((y_true_bin == 1) & (y_pred_bin == 1)).sum())
    tn = int(((y_true_bin == 0) & (y_pred_bin == 0)).sum())
    fp = int(((y_true_bin == 0) & (y_pred_bin == 1)).sum())
    fn = int(((y_true_bin == 1) & (y_pred_bin == 0)).sum())
    out = {
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "ppv": tp / max(tp + fp, 1),
        "npv": tn / max(tn + fn, 1),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }
    if prob is not None and len(np.unique(y_true_bin)) == 2:
        out["auc"] = float(roc_auc_score(y_true_bin, prob))
    return out


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> dict:
    """All scalar metrics in one JSON-friendly dictionary."""
    labels = list(range(len(C.CLASS_NAMES)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    metrics = {
        "n_test": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p.mean()),
        "macro_recall": float(r.mean()),
        "macro_f1": float(f.mean()),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "quadratic_weighted_kappa": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "per_class": {
            C.CLASS_NAMES[i]: {"precision": float(p[i]), "recall": float(r[i]),
                               "f1": float(f[i]), "support": int(s[i])}
            for i in labels
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    # clinically meaningful binary views ---------------------------------
    ref_prob = y_prob[:, C.REFERABLE_THRESHOLD:].sum(axis=1) if y_prob is not None else None
    any_prob = y_prob[:, 1:].sum(axis=1) if y_prob is not None else None
    ref = _binary_stats((y_true >= C.REFERABLE_THRESHOLD).astype(int),
                        (y_pred >= C.REFERABLE_THRESHOLD).astype(int), ref_prob)
    any_dr = _binary_stats((y_true >= 1).astype(int), (y_pred >= 1).astype(int), any_prob)
    metrics["referable_dr"] = ref
    metrics["any_dr"] = any_dr
    metrics["referable_sensitivity"] = ref["sensitivity"]
    metrics["referable_specificity"] = ref["specificity"]
    if y_prob is not None:
        try:
            metrics["macro_auc_ovr"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        except ValueError:
            pass
    # mean absolute grade error - how far off are we on the ordinal scale?
    metrics["mean_absolute_grade_error"] = float(np.abs(y_true - y_pred).mean())
    metrics["within_one_grade_accuracy"] = float((np.abs(y_true - y_pred) <= 1).mean())
    return metrics


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_training_curves(hist: pd.DataFrame, path: Path, title: str = "") -> None:
    """Accuracy and loss vs. epoch, with the phase-1 / phase-2 boundary marked."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    boundary = int((hist["phase"] == 1).sum()) + 0.5 if "phase" in hist else None
    for ax, key, label in ((axes[0], "accuracy", "Accuracy"), (axes[1], "loss", "Loss")):
        ax.plot(hist["epoch"], hist[key], marker="o", ms=3, label=f"train {label.lower()}", color="#264653")
        ax.plot(hist["epoch"], hist[f"val_{key}"], marker="s", ms=3, label=f"validation {label.lower()}", color="#e76f51")
        if boundary and boundary < hist["epoch"].max():
            ax.axvline(boundary, ls="--", color="grey", lw=1)
            ax.text(boundary + 0.2, ax.get_ylim()[1] * 0.98, "fine-tuning starts", va="top", fontsize=8, color="grey")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(f"{label} curve")
        ax.legend()
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, path: Path, normalize: bool = False, title: str = "") -> None:
    cm = np.asarray(cm)
    if normalize:
        cm_disp = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        fmt, vmax = ".2f", 1.0
    else:
        cm_disp, fmt, vmax = cm, "d", None
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(cm_disp, annot=True, fmt=fmt, cmap="Blues", vmax=vmax, cbar=True,
                xticklabels=C.CLASS_LABELS, yticklabels=C.CLASS_LABELS, ax=ax)
    ax.set_xlabel("Predicted grade")
    ax.set_ylabel("True grade")
    ax.set_title(title or ("Normalised confusion matrix" if normalize else "Confusion matrix"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_metrics(metrics: dict, path: Path) -> None:
    """Grouped bar chart of precision / recall / F1 per class."""
    pc = metrics["per_class"]
    df = pd.DataFrame({
        "class": [C.CLASS_LABELS[C.CLASS_NAMES.index(k)] for k in pc],
        "precision": [v["precision"] for v in pc.values()],
        "recall": [v["recall"] for v in pc.values()],
        "f1": [v["f1"] for v in pc.values()],
    }).melt(id_vars="class", var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=df, x="class", y="score", hue="metric", ax=ax, palette=["#264653", "#2a9d8f", "#e9c46a"])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_title("Per-class precision, recall and F1-score (test set)")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=7, padding=1)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(y_true: np.ndarray, y_prob: np.ndarray, path: Path) -> Dict[str, float]:
    """One-vs-rest ROC curve per class plus the referable-DR curve."""
    fig, ax = plt.subplots(figsize=(6, 5.5))
    aucs = {}
    for i, name in enumerate(C.CLASS_LABELS):
        y_bin = (y_true == i).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
        auc = roc_auc_score(y_bin, y_prob[:, i])
        aucs[name] = float(auc)
        ax.plot(fpr, tpr, color=PALETTE[i], lw=1.6, label=f"{name} (AUC = {auc:.3f})")
    y_ref = (y_true >= C.REFERABLE_THRESHOLD).astype(int)
    if 0 < y_ref.sum() < len(y_ref):
        p_ref = y_prob[:, C.REFERABLE_THRESHOLD:].sum(axis=1)
        fpr, tpr, _ = roc_curve(y_ref, p_ref)
        auc = roc_auc_score(y_ref, p_ref)
        aucs["Referable DR"] = float(auc)
        ax.plot(fpr, tpr, color="black", lw=2.2, ls="--", label=f"Referable DR (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="lightgrey", ls=":")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("ROC curves (one-vs-rest)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return aucs


def plot_misclassified(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray,
                       path: Path, n: int = 12) -> None:
    """Show the ``n`` most confident *wrong* predictions for error analysis."""
    wrong = np.where(y_true != y_pred)[0]
    if wrong.size == 0:
        return
    conf = y_prob[wrong, y_pred[wrong]]
    order = wrong[np.argsort(-conf)][:n]
    cols = 6
    rows = int(np.ceil(len(order) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.9 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, i in zip(axes, order):
        ax.imshow(X[i])
        ax.set_title(f"true: {C.CLASS_LABELS[y_true[i]]}\npred: {C.CLASS_LABELS[y_pred[i]]} ({y_prob[i, y_pred[i]]:.2f})",
                     fontsize=8, color="#9b2226" if abs(y_true[i] - y_pred[i]) > 1 else "black")
    fig.suptitle("Most confident misclassifications (red = off by more than one grade)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_distribution(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    """True vs predicted class counts - shows whether the model is biased to the majority class."""
    df = pd.DataFrame({
        "class": C.CLASS_LABELS * 2,
        "count": [int((y_true == i).sum()) for i in range(5)] + [int((y_pred == i).sum()) for i in range(5)],
        "which": ["true"] * 5 + ["predicted"] * 5,
    })
    fig, ax = plt.subplots(figsize=(7, 3.8))
    sns.barplot(data=df, x="class", y="count", hue="which", ax=ax, palette=["#264653", "#e9c46a"])
    ax.set_xlabel("")
    ax.set_title("True vs predicted grade distribution (test set)")
    for container in ax.containers:
        ax.bar_label(container, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, run_dir: Path,
                   batch_size: int = 32, make_figures: bool = True, prefix: str = "") -> dict:
    """Predict on the test set, compute every metric and write figures + files."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    y_prob = predict_probs(model, X_test, batch_size)
    y_pred = y_prob.argmax(axis=1)

    metrics = compute_metrics(y_test, y_pred, y_prob)
    report = classification_report(y_test, y_pred, labels=list(range(5)),
                                   target_names=C.CLASS_LABELS, zero_division=0, digits=4)
    metrics["classification_report"] = report
    with open(run_dir / f"{prefix}classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    pd.DataFrame({"y_true": y_test, "y_pred": y_pred,
                  **{f"p_{C.CLASS_NAMES[i]}": y_prob[:, i] for i in range(5)}}
                 ).to_csv(run_dir / f"{prefix}test_predictions.csv", index=False)

    if make_figures:
        cm = np.array(metrics["confusion_matrix"])
        plot_confusion_matrix(cm, run_dir / f"{prefix}confusion_matrix.png")
        plot_confusion_matrix(cm, run_dir / f"{prefix}confusion_matrix_normalised.png", normalize=True)
        plot_per_class_metrics(metrics, run_dir / f"{prefix}per_class_metrics.png")
        metrics["roc_auc"] = plot_roc_curves(y_test, y_prob, run_dir / f"{prefix}roc_curves.png")
        plot_misclassified(X_test, y_test, y_pred, y_prob, run_dir / f"{prefix}misclassified.png")
        plot_prediction_distribution(y_test, y_pred, run_dir / f"{prefix}prediction_distribution.png")
    print(report)
    return metrics

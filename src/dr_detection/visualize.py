"""
Exploratory data analysis and pipeline figures (CPU only, no TensorFlow needed
except for the augmentation grid).

Every function writes one PNG and is called by ``scripts/make_figures.py`` and
by the Colab notebook, so the report figures are fully reproducible.
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

from . import config as C  # noqa: E402
from .preprocessing import load_image, preprocess_image, preprocessing_stages  # noqa: E402

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
PALETTE = ["#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#9b2226"]


def plot_class_distribution(df: pd.DataFrame, path: Path, title: str = "Class distribution (all images)") -> None:
    counts = df["label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bars = ax.bar([C.CLASS_LABELS[i] for i in counts.index], counts.values, color=PALETTE)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 15, f"{v}\n({100*v/len(df):.1f}%)", ha="center", fontsize=9)
    ax.set_ylabel("Number of images")
    ax.set_title(title)
    ax.set_ylim(0, counts.max() * 1.18)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_split_distribution(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, path: Path) -> None:
    """Stacked bars proving that the stratified split preserved class ratios."""
    rows = []
    for name, part in (("train", df_train), ("validation", df_val), ("test", df_test)):
        for c in range(5):
            rows.append({"split": name, "class": C.CLASS_LABELS[c], "images": int((part["label"] == c).sum())})
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=d, x="class", y="images", hue="split", ax=ax, palette=["#264653", "#2a9d8f", "#e9c46a"])
    for container in ax.containers:
        ax.bar_label(container, fontsize=7)
    ax.set_xlabel("")
    ax.set_title("Stratified 70 / 10 / 20 split per class")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sample_grid(df: pd.DataFrame, path: Path, n_per_class: int = 4, method: str = "raw",
                     size: int = 224, seed: int = 0) -> None:
    """``n_per_class`` random images for each grade (raw or preprocessed)."""
    fig, axes = plt.subplots(5, n_per_class, figsize=(2.4 * n_per_class, 2.5 * 5))
    for c in range(5):
        sample = df[df["label"] == c].sample(n_per_class, random_state=seed)
        for j, (_, row) in enumerate(sample.iterrows()):
            ax = axes[c, j]
            ax.imshow(preprocess_image(load_image(row["image_path"]), method=method, size=size))
            ax.axis("off")
            if j == 0:
                ax.set_title(f"Grade {c}: {C.CLASS_LABELS[c]}", fontsize=9, loc="left")
    fig.suptitle("Sample fundus images per DR grade" + ("" if method == "raw" else f" ({method})"))
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_preprocessing_stages(image_path: str, path: Path, size: int = 224) -> None:
    """One image through every preprocessing stage (before / after evidence)."""
    stages = preprocessing_stages(load_image(image_path), size=size)
    fig, axes = plt.subplots(1, len(stages), figsize=(2.6 * len(stages), 3))
    for ax, (name, img) in zip(axes, stages.items()):
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.suptitle("Preprocessing pipeline stages")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_preprocessing_comparison(df: pd.DataFrame, path: Path, size: int = 224, seed: int = 0,
                                  methods=("raw", "clahe", "ben_graham", "clahe_ben_graham")) -> None:
    """One image per grade, shown under each preprocessing method."""
    fig, axes = plt.subplots(5, len(methods), figsize=(2.6 * len(methods), 2.6 * 5))
    for c in range(5):
        row = df[df["label"] == c].sample(1, random_state=seed).iloc[0]
        img = load_image(row["image_path"])
        for j, m in enumerate(methods):
            ax = axes[c, j]
            ax.imshow(preprocess_image(img, method=m, size=size))
            ax.axis("off")
            if c == 0:
                ax.set_title(m, fontsize=10)
            if j == 0:
                ax.text(-0.08, 0.5, C.CLASS_LABELS[c], transform=ax.transAxes, rotation=90,
                        va="center", ha="right", fontsize=9)
    fig.suptitle("Effect of each preprocessing method")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_intensity_stats(df: pd.DataFrame, path: Path, n_per_class: int = 60, seed: int = 0) -> pd.DataFrame:
    """Mean brightness and contrast per class - shows camera / illumination variability.

    Returns the table that the plot was made from (also useful in the report).
    """
    rows = []
    for c in range(5):
        sample = df[df["label"] == c].sample(min(n_per_class, int((df["label"] == c).sum())), random_state=seed)
        for _, r in sample.iterrows():
            img = load_image(r["image_path"])
            gray = img.mean(axis=2)
            fg = gray[gray > 7]  # ignore black frame
            rows.append({"class": C.CLASS_LABELS[c], "mean_brightness": fg.mean(), "contrast_std": fg.std(),
                         "height": img.shape[0], "width": img.shape[1]})
    d = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    sns.boxplot(data=d, x="class", y="mean_brightness", ax=axes[0], palette=PALETTE, hue="class", legend=False)
    axes[0].set_title("Mean brightness of retina pixels")
    axes[0].set_xlabel("")
    sns.boxplot(data=d, x="class", y="contrast_std", ax=axes[1], palette=PALETTE, hue="class", legend=False)
    axes[1].set_title("Contrast (std of retina pixels)")
    axes[1].set_xlabel("")
    for ax in axes:
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.suptitle("Illumination variability across the dataset (raw images)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return d


def plot_augmentation_grid(image_path: str, path: Path, method: str = "ben_graham", size: int = 224, n: int = 7) -> None:
    """Original preprocessed image followed by ``n`` random augmentations."""
    from .augmentation import augment_examples

    img = preprocess_image(load_image(image_path), method=method, size=size)
    variants = augment_examples(img, n=n, seed=1)
    fig, axes = plt.subplots(1, n + 1, figsize=(2.4 * (n + 1), 2.8))
    axes[0].imshow(img)
    axes[0].set_title("original", fontsize=9)
    for i, ax in enumerate(axes[1:]):
        ax.imshow(variants[i])
        ax.set_title(f"aug {i+1}", fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.suptitle("On-the-fly augmentation: flips, rotation, zoom, shift, brightness, contrast")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_class_balance(y_train: np.ndarray, weights: Dict[int, float], path: Path,
                       y_after: Optional[np.ndarray] = None) -> None:
    """Training-set class counts with the class weight annotated above each bar."""
    counts = [int((y_train == c).sum()) for c in range(5)]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(5)
    if y_after is not None:
        after = [int((y_after == c).sum()) for c in range(5)]
        ax.bar(x - 0.2, counts, width=0.4, color="#264653", label="before over-sampling")
        ax.bar(x + 0.2, after, width=0.4, color="#e9c46a", label="after over-sampling")
        ax.legend()
    else:
        ax.bar(x, counts, color=PALETTE)
    for i, c in enumerate(counts):
        ax.text(i, max(counts) * 1.03, f"w = {weights.get(i, 1):.2f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(C.CLASS_LABELS)
    ax.set_ylabel("Training images")
    ax.set_ylim(0, max(counts) * 1.18)
    ax.set_title("Training-set imbalance and the class weights used in the loss")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_experiment_comparison(results: pd.DataFrame, path: Path,
                               metrics=("accuracy", "macro_f1", "quadratic_weighted_kappa", "referable_sensitivity")) -> None:
    """Bar chart comparing several runs (backbones / preprocessing methods)."""
    d = results.melt(id_vars="run_name", value_vars=list(metrics), var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(results)), 4.2))
    sns.barplot(data=d, x="run_name", y="score", hue="metric", ax=ax,
                palette=["#264653", "#2a9d8f", "#e9c46a", "#e76f51"])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.set_title("Experiment comparison (test set)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=7, padding=1)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

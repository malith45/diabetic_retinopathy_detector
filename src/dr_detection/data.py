"""
Dataset discovery, label mapping, stratified splitting and tf.data pipelines.

Design decisions
----------------
* The dataset is small enough (3,662 images at 224 px ~ 550 MB as uint8) to be
  **preprocessed once and kept in RAM** as numpy arrays.  This makes training
  I/O-free and guarantees the exact same preprocessing for every epoch.
* Preprocessed arrays are cached to ``cache_dir`` as ``.npz`` files keyed by
  preprocessing method and image size, so re-running an experiment is instant.
* Splitting is **stratified** on the label and seeded, so every experiment
  (different backbones / preprocessing) is evaluated on the *same* test images.
* The test split is created first and never touched again; validation is then
  carved out of the remaining training images (70 / 10 / 20 overall).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from . import config as C
from .preprocessing import load_image, preprocess_image

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


# --------------------------------------------------------------------------- #
# DataFrame creation
# --------------------------------------------------------------------------- #
def normalise_label(label: str) -> Optional[int]:
    """Map any label spelling (``'Proliferate_DR'``, ``'No DR'``, ``'3'`` ...) to 0-4."""
    key = str(label).strip().lower().replace("-", "_")
    if key in C.LABEL_TO_INDEX:
        return C.LABEL_TO_INDEX[key]
    key2 = key.replace("_", " ")
    if key2 in C.LABEL_TO_INDEX:
        return C.LABEL_TO_INDEX[key2]
    try:
        val = int(float(key))
        return val if 0 <= val <= 4 else None
    except ValueError:
        return None


def build_dataframe(images_dir: str | Path) -> pd.DataFrame:
    """Scan ``images_dir/<class_folder>/*.png`` into a tidy DataFrame.

    Columns: ``image_path``, ``filename``, ``id_code``, ``folder``,
    ``label`` (0-4) and ``class_name``.

    Unlike a naive ``{'No_DR': 0, ...}`` dictionary, :func:`normalise_label`
    understands the mis-spelt ``Proliferate_DR`` folder of the Kaggle dataset,
    so **no images are silently dropped**.
    """
    images_dir = Path(images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(
            f"{images_dir} does not exist. Run scripts/download_data.py first."
        )
    rows = []
    for class_dir in sorted(p for p in images_dir.iterdir() if p.is_dir()):
        label = normalise_label(class_dir.name)
        if label is None:
            print(f"[data] WARNING: folder {class_dir.name!r} is not a known class - skipped")
            continue
        for f in sorted(class_dir.iterdir()):
            if f.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append({
                    "image_path": str(f),
                    "filename": f.name,
                    "id_code": f.stem,
                    "folder": class_dir.name,
                    "label": label,
                    "class_name": C.CLASS_NAMES[label],
                })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No images found under {images_dir}")
    return df


def cross_check_with_csv(df: pd.DataFrame, data_dir: str | Path) -> pd.DataFrame:
    """If the dataset ships a ``train.csv`` (id_code, diagnosis) verify labels.

    Returns the DataFrame with an extra ``csv_diagnosis`` column and prints the
    number of disagreements (expected: 0).  This is a sanity check that the
    folder structure and the official label file agree.
    """
    csv_path = None
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower() == "train.csv":
                csv_path = Path(root) / f
                break
    if csv_path is None:
        return df
    csv = pd.read_csv(csv_path)
    if {"id_code", "diagnosis"} <= set(csv.columns):
        merged = df.merge(csv[["id_code", "diagnosis"]], on="id_code", how="left")
        merged = merged.rename(columns={"diagnosis": "csv_diagnosis"})
        mismatches = int((merged["csv_diagnosis"].notna() & (merged["csv_diagnosis"] != merged["label"])).sum())
        missing = int(merged["csv_diagnosis"].isna().sum())
        print(f"[data] train.csv cross-check: {mismatches} label mismatches, {missing} ids not in csv")
        return merged
    return df


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Return a table with count and percentage per class (for the report)."""
    counts = df["label"].value_counts().sort_index()
    table = pd.DataFrame({
        "grade": counts.index,
        "class": [C.CLASS_LABELS[i] for i in counts.index],
        "images": counts.values,
        "percent": (100 * counts.values / len(df)).round(1),
    })
    return table.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def split_dataframe(
    df: pd.DataFrame,
    val_frac: float = 0.10,
    test_frac: float = 0.20,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train / validation / test split (defaults to 70 / 10 / 20)."""
    train_val, test = train_test_split(
        df, test_size=test_frac, stratify=df["label"], random_state=seed
    )
    # validation fraction relative to the remaining data
    rel_val = val_frac / (1.0 - test_frac)
    train, val = train_test_split(
        train_val, test_size=rel_val, stratify=train_val["label"], random_state=seed
    )
    for name, part in (("train", train), ("val", val), ("test", test)):
        part = part.copy()
        part["split"] = name
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def subsample_per_class(df: pd.DataFrame, n_per_class: int, seed: int = 42) -> pd.DataFrame:
    """Keep at most ``n_per_class`` images of every class (used by quick tests)."""
    parts = [
        g.sample(min(len(g), n_per_class), random_state=seed)
        for _, g in df.groupby("label")
    ]
    return pd.concat(parts).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Image loading with cache
# --------------------------------------------------------------------------- #
def load_images(
    df: pd.DataFrame,
    method: str,
    size: int,
    cache_dir: Optional[str | Path] = None,
    cache_tag: str = "",
    desc: str = "images",
) -> np.ndarray:
    """Load + preprocess every image of ``df`` into a ``uint8 (N, size, size, 3)`` array.

    If ``cache_dir`` is given the result is stored as ``.npz`` and re-used on the
    next call with the same ``method``, ``size``, ``cache_tag`` and number of
    rows (the cache also stores the id codes to detect a mismatch).
    """
    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{cache_tag}_{method}_{size}_{len(df)}.npz"
        if cache_path.exists():
            data = np.load(cache_path, allow_pickle=False)
            if list(data["ids"]) == list(df["id_code"]):
                return data["X"]

    X = np.empty((len(df), size, size, 3), dtype=np.uint8)
    for i, path in enumerate(tqdm(df["image_path"], desc=f"Preprocessing {desc}", leave=False)):
        X[i] = preprocess_image(load_image(path), method=method, size=size)

    if cache_path is not None:
        np.savez(cache_path, X=X, ids=np.array(df["id_code"], dtype=str))
    return X


# --------------------------------------------------------------------------- #
# tf.data pipelines
# --------------------------------------------------------------------------- #
def make_tf_dataset(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = False,
    augmentation=None,
    seed: int = 42,
):
    """Wrap numpy arrays into a batched, prefetched ``tf.data.Dataset``.

    ``augmentation`` is an optional Keras ``Sequential`` of preprocessing layers
    that is applied *per batch on the fly* (training set only).  Images are
    passed to the model as float32 in the 0-255 range; each backbone applies
    its own normalisation inside the model (see ``model.py``).
    """
    import tensorflow as tf  # imported lazily so figure scripts stay TF-free

    ds = tf.data.Dataset.from_tensor_slices((X, y.astype(np.int32)))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(X), 4096), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.map(lambda img, lbl: (tf.cast(img, tf.float32), lbl), num_parallel_calls=tf.data.AUTOTUNE)
    if augmentation is not None:
        ds = ds.map(lambda img, lbl: (augmentation(img, training=True), lbl),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def prepare_data(cfg: C.Config, verbose: bool = True) -> Dict[str, object]:
    """One call that returns everything the training loop needs.

    Returns a dict with the three DataFrames (``df_train`` ...), the preprocessed
    arrays (``X_train`` ...) and label vectors (``y_train`` ...).
    """
    df = build_dataframe(cfg.images_dir)
    df = cross_check_with_csv(df, cfg.data_dir)
    if cfg.quick_test:
        df = subsample_per_class(df, cfg.quick_test_per_class, cfg.seed)
    df_train, df_val, df_test = split_dataframe(df, cfg.val_frac, cfg.test_frac, cfg.seed)

    if verbose:
        print(f"[data] total={len(df)}  train={len(df_train)}  val={len(df_val)}  test={len(df_test)}")

    tag = "quick" if cfg.quick_test else "full"
    out = {"df_all": df, "df_train": df_train, "df_val": df_val, "df_test": df_test}
    for name, part in (("train", df_train), ("val", df_val), ("test", df_test)):
        out[f"X_{name}"] = load_images(
            part, cfg.preprocess, cfg.img_size, cfg.cache_dir, cache_tag=f"{tag}_{name}", desc=name
        )
        out[f"y_{name}"] = part["label"].to_numpy()
    return out

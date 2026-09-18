"""
Data augmentation and class balancing.

Augmentation
------------
Fundus photographs are rotation- and flip-invariant (a rotated retina is still
the same retina) and vary in brightness / contrast between cameras.  We
therefore apply, *on the fly and only to the training set*:

* random horizontal + vertical flips
* random rotation (+/- 36 degrees)
* random zoom (+/- 10 %)
* random translation (+/- 5 %)
* random brightness (+/- 10 %)
* random contrast (+/- 10 %)

Because the augmentation is implemented with Keras preprocessing layers it runs
on the GPU and produces a *different* variant of every image at every epoch,
which is far more effective against over-fitting than a fixed, pre-generated
set of augmented copies.

Class balancing
---------------
The dataset is heavily imbalanced (49 % No DR vs 5 % Severe).  Two strategies
are provided:

* :func:`compute_class_weights` - inverse-frequency weights passed to
  ``model.fit(class_weight=...)`` (default; lecturer's recommended approach)
* :func:`oversample_minority`   - random over-sampling of the minority classes
  in the training split only (never in validation / test)
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from . import config as C


# --------------------------------------------------------------------------- #
# Augmentation pipeline
# --------------------------------------------------------------------------- #
def build_augmentation(seed: int = 42):
    """Return a Keras ``Sequential`` of random augmentation layers.

    The layers are only active when called with ``training=True`` (which
    ``data.make_tf_dataset`` does), so the *same* object can be shown in a
    model summary without affecting inference.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical", seed=seed),
            layers.RandomRotation(0.10, fill_mode="constant", fill_value=0.0, seed=seed),
            layers.RandomZoom(0.10, fill_mode="constant", fill_value=0.0, seed=seed),
            layers.RandomTranslation(0.05, 0.05, fill_mode="constant", fill_value=0.0, seed=seed),
            layers.RandomBrightness(0.10, value_range=(0.0, 255.0), seed=seed),
            layers.RandomContrast(0.10, seed=seed),
        ],
        name="augmentation",
    )


def augment_examples(img: np.ndarray, n: int = 8, seed: int = 0) -> np.ndarray:
    """Return ``n`` augmented variants of one uint8 image (for figures)."""
    import tensorflow as tf

    aug = build_augmentation(seed=seed)
    batch = tf.repeat(tf.cast(img[None, ...], tf.float32), n, axis=0)
    out = aug(batch, training=True).numpy()
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Class balancing
# --------------------------------------------------------------------------- #
def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Inverse-frequency ("balanced") class weights, as a ``{class: weight}`` dict.

    ``weight_c = N / (num_classes * n_c)``  - rare classes get a weight > 1,
    frequent classes a weight < 1, so every class contributes roughly equally
    to the loss.
    """
    classes = np.arange(C.Config().num_classes)
    present = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=present, y=y)
    out = {int(c): 1.0 for c in classes}
    out.update({int(c): float(w) for c, w in zip(present, weights)})
    return out


def oversample_minority(X: np.ndarray, y: np.ndarray, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Random over-sampling so that every class has as many samples as the largest.

    Only ever apply this to the **training** split.  Because augmentation is
    applied on the fly, duplicated images still receive different random
    transformations, so this is *not* simple duplication of pixels.
    """
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    target = counts.max()
    idx = []
    for c, n in zip(classes, counts):
        cls_idx = np.where(y == c)[0]
        extra = rng.choice(cls_idx, size=target - n, replace=True) if n < target else np.array([], dtype=int)
        idx.append(np.concatenate([cls_idx, extra]))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return X[idx], y[idx]


def balance_summary(y_before: np.ndarray, y_after: np.ndarray | None, weights: Dict[int, float]):
    """Small table (list of dicts) describing the balancing applied - for the report."""
    rows = []
    for c in range(len(C.CLASS_NAMES)):
        rows.append({
            "grade": c,
            "class": C.CLASS_LABELS[c],
            "train_images": int((y_before == c).sum()),
            "after_oversampling": int((y_after == c).sum()) if y_after is not None else int((y_before == c).sum()),
            "class_weight": round(weights.get(c, 1.0), 3),
        })
    return rows

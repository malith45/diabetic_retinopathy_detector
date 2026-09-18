"""
Grad-CAM (Gradient-weighted Class Activation Mapping) explainability.

Grad-CAM (Selvaraju et al., 2017) answers the question *"which part of the
retina made the network choose this grade?"*.  It back-propagates the score of
the predicted class to the last convolutional feature map of the backbone,
averages the gradients per channel to obtain importance weights, and builds a
weighted sum of the feature maps.  High values mark the regions that increased
the class score - ideally lesions such as haemorrhages or exudates rather than
the image border.

Implementation note
-------------------
Our classifier is a functional model of the form
``input -> normalisation -> backbone (nested model) -> head layers``.  Because
``include_top=False`` makes the backbone's *output* equal to its last
convolutional activation, we can split the forward pass into
``pre -> backbone -> post`` and take gradients w.r.t. the backbone output.  This
avoids the "graph disconnected" error that occurs when trying to build a
sub-model from a layer inside a nested Keras 3 model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import config as C  # noqa: E402
from .model import get_backbone  # noqa: E402


def make_gradcam_heatmap(model, img: np.ndarray, pred_index: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, int]:
    """Compute the Grad-CAM heat-map for one image.

    Parameters
    ----------
    model      : classifier built by ``model.build_model`` (or loaded from disk)
    img        : uint8 / float RGB image ``(H, W, 3)`` already preprocessed
    pred_index : class to explain; defaults to the predicted class

    Returns
    -------
    heatmap : float32 array in 0-1 with the backbone's spatial resolution (e.g. 7x7)
    probs   : the softmax probabilities of the image
    index   : the class index that was explained
    """
    import tensorflow as tf

    base = get_backbone(model)
    layers_list = list(model.layers)
    idx = layers_list.index(base)
    pre_layers = [l for l in layers_list[1:idx]]     # skip InputLayer
    post_layers = layers_list[idx + 1:]

    x = tf.convert_to_tensor(img[None, ...].astype(np.float32))
    for layer in pre_layers:
        x = layer(x)

    with tf.GradientTape() as tape:
        conv = base(x, training=False)
        tape.watch(conv)
        h = conv
        for layer in post_layers:
            h = layer(h)
        probs = h
        if pred_index is None:
            pred_index = int(tf.argmax(probs[0]))
        score = probs[:, pred_index]

    grads = tape.gradient(score, conv)                     # (1, h, w, c)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))         # (c,)
    heatmap = tf.reduce_sum(conv[0] * pooled, axis=-1)     # (h, w)
    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy().astype(np.float32), probs.numpy()[0], int(pred_index)


def overlay_heatmap(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a JET-coloured heat-map over the RGB image (returns uint8 RGB)."""
    h, w = img.shape[:2]
    hm = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    hm = np.uint8(255 * np.clip(hm, 0, 1))
    coloured = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    coloured = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)
    out = cv2.addWeighted(img.astype(np.uint8), 1 - alpha, coloured, alpha, 0)
    return out


def gradcam_grid(model, X: np.ndarray, y: np.ndarray, path: Path, n_per_class: int = 2, seed: int = 0) -> None:
    """Figure: for each class, ``n_per_class`` test images with their Grad-CAM overlay."""
    rng = np.random.default_rng(seed)
    n_classes = len(C.CLASS_NAMES)
    fig, axes = plt.subplots(n_classes, 2 * n_per_class, figsize=(2.6 * 2 * n_per_class, 2.6 * n_classes))
    for c in range(n_classes):
        idx = np.where(y == c)[0]
        pick = rng.choice(idx, size=min(n_per_class, len(idx)), replace=False) if len(idx) else []
        for j in range(n_per_class):
            ax_img, ax_cam = axes[c, 2 * j], axes[c, 2 * j + 1]
            ax_img.axis("off")
            ax_cam.axis("off")
            if j >= len(pick):
                continue
            i = pick[j]
            heatmap, probs, k = make_gradcam_heatmap(model, X[i])
            ax_img.imshow(X[i])
            ax_img.set_title(f"true: {C.CLASS_LABELS[c]}", fontsize=8)
            ax_cam.imshow(overlay_heatmap(X[i], heatmap))
            ax_cam.set_title(f"pred: {C.CLASS_LABELS[k]} ({probs[k]:.2f})", fontsize=8,
                             color="black" if k == c else "#9b2226")
    fig.suptitle("Grad-CAM: regions that drove the prediction (red = most important)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

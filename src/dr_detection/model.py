"""
CNN architectures with transfer learning.

Architecture (identical for every backbone)::

    Input (224, 224, 3)  - float32, 0-255
      |
    Backbone-specific normalisation  (built into the graph so the saved model
      |                               accepts raw images)
    ImageNet-pretrained backbone, include_top=False   (frozen in phase 1)
      |
    GlobalAveragePooling2D
      |
    Dropout(p) -> Dense(512, ReLU) -> Dropout(p) -> Dense(256, ReLU)
      |
    Dense(5, softmax)      -> probability of each DR grade

Why these backbones?
--------------------
* **EfficientNetB0 / B3** - best accuracy per FLOP; B0 trains quickly on a
  free Colab GPU and was designed with compound scaling (Tan & Le, 2019).
* **ResNet50V2**  - the classic residual network, strong baseline.
* **MobileNetV2** - depth-wise separable convolutions; tiny and fast, ideal
  for on-device screening in low-resource clinics.
* **DenseNet121** - dense connectivity, popular in medical imaging papers.

Transfer learning is applied in two phases (see ``train.py``): first only the
new head is trained with the backbone frozen, then the top ``unfreeze_layers``
layers of the backbone are un-frozen and trained with a 10x smaller learning
rate.  BatchNormalization layers stay frozen during fine-tuning - updating
their running statistics with small batches of a new domain is a well known
cause of accuracy collapse when fine-tuning EfficientNet.
"""

from __future__ import annotations

from typing import Tuple

from . import config as C


def _normalisation_layer(backbone: str):
    """Return the input normalisation layer expected by ``backbone``.

    * EfficientNet models in Keras already contain their own rescaling and
      normalisation layers, so the identity is returned.
    * ResNet50V2 / MobileNetV2 use "tf" mode: ``x / 127.5 - 1``  (-1 ... 1)
    * DenseNet121 uses "torch" mode: ``(x / 255 - mean) / std`` per channel
    """
    from tensorflow.keras import layers

    if backbone.startswith("EfficientNet"):
        return layers.Identity(name="normalisation_identity")
    if backbone in ("ResNet50V2", "MobileNetV2"):
        return layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="normalisation_tf")
    if backbone == "DenseNet121":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        scale = [1.0 / (255.0 * s) for s in std]
        offset = [-m / s for m, s in zip(mean, std)]
        return layers.Rescaling(scale=scale, offset=offset, name="normalisation_torch")
    raise ValueError(f"Unknown backbone {backbone!r}")


def _backbone_constructor(backbone: str):
    from tensorflow.keras import applications as A

    return {
        "EfficientNetB0": A.EfficientNetB0,
        "EfficientNetB3": A.EfficientNetB3,
        "ResNet50V2": A.ResNet50V2,
        "MobileNetV2": A.MobileNetV2,
        "DenseNet121": A.DenseNet121,
    }[backbone]


def build_model(cfg: C.Config, weights: str | None = "imagenet"):
    """Create the transfer-learning model described in the module docstring.

    Returns
    -------
    model : ``keras.Model`` - the full classifier (raw 0-255 images in)
    base  : ``keras.Model`` - the pretrained backbone (a layer inside ``model``)
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    constructor = _backbone_constructor(cfg.backbone)
    base = constructor(weights=weights, include_top=False, input_shape=cfg.input_shape)
    base._name = "backbone"  # stable name so the app / Grad-CAM can find it
    base.trainable = False   # phase 1: frozen feature extractor

    inputs = keras.Input(shape=cfg.input_shape, name="fundus_image")
    x = _normalisation_layer(cfg.backbone)(inputs)
    x = base(x, training=False)  # training=False keeps BatchNorm in inference mode
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(cfg.dropout, name="dropout_1")(x)
    x = layers.Dense(cfg.dense_units[0], activation="relu", name="fc_1")(x)
    x = layers.Dropout(cfg.dropout, name="dropout_2")(x)
    x = layers.Dense(cfg.dense_units[1], activation="relu", name="fc_2")(x)
    outputs = layers.Dense(cfg.num_classes, activation="softmax", name="dr_grade")(x)

    model = keras.Model(inputs, outputs, name=f"DR_{cfg.backbone}")
    return model, base


def get_backbone(model):
    """Return the nested backbone model of a classifier built by :func:`build_model`."""
    from tensorflow import keras

    for layer in model.layers:
        if isinstance(layer, keras.Model):
            return layer
    raise ValueError("No nested backbone model found")


def set_fine_tuning(base, unfreeze_layers: int) -> Tuple[int, int]:
    """Un-freeze the top ``unfreeze_layers`` layers of ``base`` (phase 2).

    BatchNormalization layers are kept frozen.  Returns the number of
    trainable and frozen layers for logging.
    """
    from tensorflow.keras import layers

    base.trainable = True
    n_layers = len(base.layers)
    cutoff = max(0, n_layers - unfreeze_layers)
    trainable = 0
    for i, layer in enumerate(base.layers):
        if i < cutoff or isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
            trainable += 1
    return trainable, n_layers - trainable


def count_params(model) -> dict:
    """Trainable / non-trainable / total parameter counts as plain ints."""
    import numpy as np

    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    non_trainable = int(sum(np.prod(w.shape) for w in model.non_trainable_weights))
    return {"trainable": trainable, "non_trainable": non_trainable, "total": trainable + non_trainable}


def compile_model(model, learning_rate: float):
    """Compile with Adam + sparse categorical cross-entropy (integer labels)."""
    from tensorflow import keras

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

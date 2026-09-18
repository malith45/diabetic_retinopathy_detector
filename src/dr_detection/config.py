"""
Central configuration for the DR stage detection project.

Everything that a user might want to tune (image size, epochs, learning
rates, backbone, preprocessing method, paths) is defined here so that the
notebook, the CLI scripts and the Streamlit app all read the same values.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

# --------------------------------------------------------------------------- #
# Dataset constants
# --------------------------------------------------------------------------- #

#: Kaggle identifier of the dataset used in this project.  It is the APTOS 2019
#: Blindness Detection training set, resized to 224x224 by the dataset author.
KAGGLE_DATASET = "sovitrath/diabetic-retinopathy-224x224-2019-data"

#: Name of the zip file the Kaggle CLI produces for the dataset above.
KAGGLE_ZIP_NAME = "diabetic-retinopathy-224x224-2019-data.zip"

#: Canonical class names (index = severity grade used by the ICDR scale).
CLASS_NAMES: List[str] = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"]

#: Human friendly labels used in figures and in the demo application.
CLASS_LABELS: List[str] = [
    "No DR",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "Proliferative DR",
]

#: Short clinical description of each grade (International Clinical DR scale).
CLASS_DESCRIPTIONS: Dict[int, str] = {
    0: "No abnormalities visible.",
    1: "Microaneurysms only.",
    2: "More than microaneurysms but less than severe NPDR "
       "(haemorrhages, hard exudates, cotton-wool spots).",
    3: "Severe non-proliferative DR: extensive haemorrhages, venous beading "
       "or IRMA (4-2-1 rule).",
    4: "Proliferative DR: neovascularisation and/or vitreous or pre-retinal "
       "haemorrhage.",
}

#: Mapping from *every* folder / label spelling we have seen in the wild to the
#: numeric grade.  The dataset on Kaggle spells the last class
#: ``Proliferate_DR`` (sic), which is why several spellings are listed.
LABEL_TO_INDEX: Dict[str, int] = {
    "no_dr": 0, "no dr": 0, "nodr": 0, "0": 0,
    "mild": 1, "mild_dr": 1, "1": 1,
    "moderate": 2, "moderate_dr": 2, "2": 2,
    "severe": 3, "severe_dr": 3, "3": 3,
    "proliferative_dr": 4, "proliferate_dr": 4, "proliferative dr": 4,
    "proliferate dr": 4, "proliferative": 4, "proliferate": 4, "pdr": 4, "4": 4,
}

#: Grades >= this value are considered *referable* DR (moderate NPDR or worse),
#: the threshold used by most screening programmes and by Gulshan et al. (2016).
REFERABLE_THRESHOLD = 2

#: Supported preprocessing methods (see ``dr_detection.preprocessing``).
PREPROCESS_METHODS = ["raw", "ben_graham", "clahe", "clahe_ben_graham", "unsharp"]

#: Supported backbones (see ``dr_detection.model``).
BACKBONES = ["EfficientNetB0", "EfficientNetB3", "ResNet50V2", "MobileNetV2", "DenseNet121"]


def project_root() -> Path:
    """Return the repository root (two levels above this file)."""
    return Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Run configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    """All hyper-parameters and paths for one experiment run.

    The defaults reproduce the *main* experiment reported in the coursework
    report: EfficientNetB0 + Ben Graham preprocessing, two-phase training.
    """

    # ---- paths -------------------------------------------------------------
    data_dir: str = str(project_root() / "data" / "diabetic_retinopathy_data")
    cache_dir: str = str(project_root() / "data" / "cache")
    results_dir: str = str(project_root() / "results")
    models_dir: str = str(project_root() / "models")

    # ---- images -------------------------------------------------------------
    img_size: int = 224
    channels: int = 3
    preprocess: str = "ben_graham"      # one of PREPROCESS_METHODS

    # ---- splits -------------------------------------------------------------
    val_frac: float = 0.10              # 70 / 10 / 20 split as in the lecturer's guide
    test_frac: float = 0.20
    seed: int = 42

    # ---- model --------------------------------------------------------------
    backbone: str = "EfficientNetB0"    # one of BACKBONES
    dropout: float = 0.3
    dense_units: tuple = (512, 256)

    # ---- training -----------------------------------------------------------
    batch_size: int = 32
    epochs_phase1: int = 10             # frozen backbone, train the head
    epochs_phase2: int = 20             # fine-tune the last `unfreeze_layers` layers
    lr_phase1: float = 1e-4
    lr_phase2: float = 1e-5
    unfreeze_layers: int = 30
    early_stopping_patience: int = 6
    reduce_lr_patience: int = 3
    use_class_weights: bool = True
    oversample_minority: bool = False   # alternative balancing strategy
    use_augmentation: bool = True

    # ---- misc ---------------------------------------------------------------
    run_name: str = ""                  # auto-generated when empty
    quick_test: bool = False            # tiny subset + 1 epoch, to smoke-test the pipeline
    quick_test_per_class: int = 24

    # ------------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.preprocess not in PREPROCESS_METHODS:
            raise ValueError(f"preprocess must be one of {PREPROCESS_METHODS}, got {self.preprocess!r}")
        if self.backbone not in BACKBONES:
            raise ValueError(f"backbone must be one of {BACKBONES}, got {self.backbone!r}")
        if not self.run_name:
            self.run_name = f"{self.backbone}_{self.preprocess}_{self.img_size}"
            if self.quick_test:
                self.run_name += "_quicktest"

    # convenience -------------------------------------------------------------
    @property
    def input_shape(self) -> tuple:
        return (self.img_size, self.img_size, self.channels)

    @property
    def num_classes(self) -> int:
        return len(CLASS_NAMES)

    @property
    def run_dir(self) -> Path:
        return Path(self.results_dir) / self.run_name

    @property
    def images_dir(self) -> Path:
        """Folder that contains one sub-folder per class."""
        for candidate in ("colored_images", "gaussian_filtered_images", "images", ""):
            p = Path(self.data_dir) / candidate if candidate else Path(self.data_dir)
            if p.exists() and any(c.is_dir() for c in p.iterdir()):
                return p
        return Path(self.data_dir) / "colored_images"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["dense_units"] = list(self.dense_units)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        d = dict(d)
        if "dense_units" in d:
            d["dense_units"] = tuple(d["dense_units"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

"""
Preprocessing pipeline of the Kaggle training notebook (Steps 6 and 11).

The deployed model was trained on images produced by exactly these functions, so the
prototype must use them unchanged; any difference (another crop, no padding, a different
filter) would present the network with images it was not trained on.

Pipeline:  crop black border -> pad to square -> resize 224 -> enhancement -> retina mask

Enhancement methods (selected experimentally in the notebook, Step 9):
    raw         no enhancement
    clahe       CLAHE on the L channel of LAB (clip limit 2.0, 8 x 8 tiles)
    enhanced    CLAHE -> Gaussian blur (sigma 0.5) -> unsharp mask (sigma 2, amount 1)
    ben_graham  4*I - 4*GaussianBlur(I, sigma 10) + 128
The noise filter and the edge operator of "enhanced" were chosen by measurement in the
notebook (Step 6); the chosen names are stored in ``model_metadata.json`` and checked below.
"""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np

IMG_SIZE = 224
METHODS = ("raw", "clahe", "enhanced", "ben_graham")


# ---- geometric standardisation -------------------------------------------------------------
def crop_to_retina(img: np.ndarray, tol: int = 10) -> np.ndarray:
    """Remove the rows and columns that contain only the black background."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return img
    return img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def pad_to_square(img: np.ndarray) -> np.ndarray:
    """Pad with black to a square so that resizing does not distort the retina."""
    h, w = img.shape[:2]
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    return cv2.copyMakeBorder(img, top, side - h - top, left, side - w - left,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))


def resize_to(img: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Area interpolation when shrinking (anti-aliased), bicubic when enlarging."""
    interp = cv2.INTER_AREA if max(img.shape[:2]) > size else cv2.INTER_CUBIC
    return cv2.resize(img, (size, size), interpolation=interp)


def standardise(img: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    return resize_to(pad_to_square(crop_to_retina(img)), size)


def retina_mask(img: np.ndarray, tol: int = 10, erode_px: int = 5) -> np.ndarray:
    """Binary (0/1) retina mask, eroded to drop the rim where filters create artefacts."""
    mask = (cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) > tol).astype(np.uint8)
    if erode_px:
        mask = cv2.erode(mask, np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8))
    return mask


def apply_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return img * mask[..., None]


# ---- enhancement ----------------------------------------------------------------------------
def clahe_rgb(img: np.ndarray, clip: float = 2.0, tiles: int = 8) -> np.ndarray:
    """CLAHE on the L (lightness) channel of LAB; colour channels are untouched."""
    l_ch, a_ch, b_ch = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2LAB))
    l_ch = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(l_ch)
    return cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2RGB)


def gaussian_denoise(img: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), sigma)


def unsharp_mask(img: np.ndarray) -> np.ndarray:
    return cv2.addWeighted(img, 2.0, cv2.GaussianBlur(img, (0, 0), 2.0), -1.0, 0)


def ben_graham(img: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    return cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigma), -4, 128)


def enhanced(img: np.ndarray) -> np.ndarray:
    return unsharp_mask(gaussian_denoise(clahe_rgb(img)))


PREPROCESSORS = {"raw": lambda im: im, "clahe": clahe_rgb, "enhanced": enhanced, "ben_graham": ben_graham}


def check_spec(spec: dict | None) -> None:
    """Fail loudly if the model was trained with filters this module does not implement."""
    if not spec:
        return
    if spec.get("noise_filter", "Gaussian s=0.5") != "Gaussian s=0.5" or \
            spec.get("edge_operator", "Unsharp mask") != "Unsharp mask":
        raise ValueError(f"Model trained with a different 'enhanced' pipeline: {spec}")


def preprocess(img_rgb: np.ndarray, method: str, size: int = IMG_SIZE) -> np.ndarray:
    """Standardise geometry, apply one enhancement method and mask the background."""
    std = standardise(img_rgb, size)
    return apply_mask(PREPROCESSORS[method](std), retina_mask(std))


def stages(img_rgb: np.ndarray, size: int = IMG_SIZE) -> Dict[str, np.ndarray]:
    """Every intermediate result, for the prototype's "preprocessing pipeline" panel."""
    cropped = crop_to_retina(img_rgb)
    std = standardise(img_rgb, size)
    mask = retina_mask(std)
    c = clahe_rgb(std)
    d = gaussian_denoise(c)
    return {
        "1. Original": img_rgb,
        "2. Crop border": cropped,
        "3. Pad + resize": std,
        "4. CLAHE": apply_mask(c, mask),
        "5. Denoise": apply_mask(d, mask),
        "6. Unsharp": apply_mask(unsharp_mask(d), mask),
        "Ben Graham": apply_mask(ben_graham(std), mask),
    }

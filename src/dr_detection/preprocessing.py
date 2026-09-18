"""
Image preprocessing for colour fundus photographs.

Every function here takes and returns **RGB uint8 numpy arrays** so that the
same code can be used by the training pipeline (numpy / tf.data), by the
figure scripts and by the Streamlit demo application.

Pipeline (in the order applied by :func:`preprocess_image`):

1. **Black-border cropping** - fundus photographs contain a large black frame
   around the circular retina.  Cropping removes uninformative pixels and
   makes the retina fill the frame consistently.
2. **Resizing** to a square ``img_size`` so that all images share one shape.
3. **Contrast / edge enhancement** using one of

   * ``raw``              - nothing further (baseline)
   * ``ben_graham``       - Gaussian-blur subtraction (Graham, 2015).  Removes
                            illumination differences between cameras and
                            sharpens vessels, haemorrhages and exudates.
   * ``clahe``            - Contrast Limited Adaptive Histogram Equalisation on
                            the L channel of the LAB colour space.
   * ``clahe_ben_graham`` - CLAHE followed by Ben Graham.
   * ``unsharp``          - unsharp masking (edge enhancement).

Noise removal is achieved implicitly by the Gaussian component of Ben Graham
and by the median filter in :func:`denoise`.
"""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_image(path: str) -> np.ndarray:
    """Read an image from disk and return it as **RGB uint8**.

    OpenCV loads images as BGR, so the channels are swapped here once and the
    rest of the code base can assume RGB everywhere.
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- #
# Individual steps
# --------------------------------------------------------------------------- #
def crop_black_border(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """Crop the black frame around the retina.

    A pixel is considered "background" when its grey level is <= ``tol``.
    Rows and columns that contain *only* background are removed.  If the mask
    is empty (completely black image) the image is returned unchanged.
    """
    if img.ndim == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return img
    return img[rows[0]: rows[-1] + 1, cols[0]: cols[-1] + 1]


def resize(img: np.ndarray, size: int) -> np.ndarray:
    """Resize to ``size x size`` using area interpolation (good for shrinking)."""
    interp = cv2.INTER_AREA if img.shape[0] > size else cv2.INTER_CUBIC
    return cv2.resize(img, (size, size), interpolation=interp)


def ben_graham(img: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """Ben Graham's preprocessing (winner of the 2015 Kaggle DR competition).

    ``out = 4 * img - 4 * GaussianBlur(img) + 128``

    Subtracting a heavily blurred copy removes slow illumination gradients and
    colour casts, while the factor 4 amplifies local contrast, so that fine
    lesions (micro-aneurysms, haemorrhages) stand out.  The +128 offset keeps
    the result centred in the 0-255 range.
    """
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    return cv2.addWeighted(img, 4, blurred, -4, 128)


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalisation.

    CLAHE is applied to the *lightness* channel only (LAB colour space) so that
    colours are preserved while local contrast is boosted.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge((l_channel, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def unsharp_mask(img: np.ndarray, sigma: float = 3.0, amount: float = 1.5) -> np.ndarray:
    """Edge enhancement by unsharp masking: ``img + amount * (img - blur)``."""
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def denoise(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Light median filtering to suppress sensor noise without blurring edges."""
    return cv2.medianBlur(img, ksize)


def circular_mask(img: np.ndarray, radius_frac: float = 0.98) -> np.ndarray:
    """Zero-out everything outside a centred circle.

    Ben Graham preprocessing turns the black frame grey; masking it back to
    black removes that artificial border so the network cannot use it.
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), int(min(h, w) // 2 * radius_frac), 255, -1)
    return cv2.bitwise_and(img, img, mask=mask)


def edge_map(img: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    """Canny edge map of the green channel (vessels are most visible in green).

    Used only for visualisation / EDA - it shows what "edge enhancement"
    actually emphasises in a fundus image.
    """
    green = img[:, :, 1] if img.ndim == 3 else img
    return cv2.Canny(green, low, high)


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #
def preprocess_image(img: np.ndarray, method: str = "ben_graham", size: int = 224) -> np.ndarray:
    """Apply the complete preprocessing pipeline to one RGB uint8 image.

    Parameters
    ----------
    img    : RGB uint8 array of any size
    method : one of ``config.PREPROCESS_METHODS``
    size   : output side length in pixels

    Returns
    -------
    RGB uint8 array of shape ``(size, size, 3)``.
    """
    img = crop_black_border(img)
    img = resize(img, size)

    if method == "raw":
        return img
    if method == "ben_graham":
        return circular_mask(ben_graham(img))
    if method == "clahe":
        return apply_clahe(img)
    if method == "clahe_ben_graham":
        return circular_mask(ben_graham(apply_clahe(img)))
    if method == "unsharp":
        return unsharp_mask(img)
    raise ValueError(f"Unknown preprocessing method: {method!r}")


def preprocessing_stages(img: np.ndarray, size: int = 224) -> Dict[str, np.ndarray]:
    """Return every intermediate result, for the "before / after" figure."""
    cropped = crop_black_border(img)
    resized = resize(cropped, size)
    stages = {
        "1. Original": img,
        "2. Border crop": cropped,
        "3. Resize": resized,
        "4. CLAHE": apply_clahe(resized),
        "5. Ben Graham": circular_mask(ben_graham(resized)),
        "6. Unsharp mask": unsharp_mask(resized),
        "7. Edge map": edge_map(resized),
    }
    return stages

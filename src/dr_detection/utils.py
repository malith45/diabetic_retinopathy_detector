"""Small helpers: seeding, JSON I/O, timing and hardware info."""

from __future__ import annotations

import json
import os
import platform
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and TensorFlow for reproducible experiments."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        try:
            from tensorflow import keras

            keras.utils.set_random_seed(seed)
        except Exception:
            pass
    except ImportError:
        pass


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


class _NumpyEncoder(json.JSONEncoder):
    """Make numpy scalars / arrays JSON serialisable."""

    def default(self, o: Any):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def save_json(obj: Any, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, cls=_NumpyEncoder)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@contextmanager
def timer(label: str = ""):
    """``with timer('training'): ...`` prints the elapsed time."""
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    print(f"[timer] {label}: {dt/60:.1f} min ({dt:.0f} s)")


def hardware_info() -> dict:
    """Describe the machine the experiment ran on (goes into metrics.json)."""
    info = {"platform": platform.platform(), "python": platform.python_version()}
    try:
        import tensorflow as tf

        info["tensorflow"] = tf.__version__
        gpus = tf.config.list_physical_devices("GPU")
        info["gpu"] = [g.name for g in gpus]
        try:
            details = tf.config.experimental.get_device_details(gpus[0]) if gpus else {}
            info["gpu_name"] = details.get("device_name", "")
        except Exception:
            pass
    except ImportError:
        info["tensorflow"] = None
    return info

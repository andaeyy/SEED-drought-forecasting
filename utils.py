from __future__ import annotations

import math
import numpy as np


def enable_tf_gpu_memory_growth():
    try:
        from config import GPU_DEVICE_ID  # noqa: F401 - configures CUDA before TensorFlow import
        import tensorflow as tf

        for g in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass


def _erf(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    try:
        from scipy.special import erf as sp_erf

        return sp_erf(x).astype(np.float32)
    except Exception:
        erf_vec = np.vectorize(math.erf, otypes=[np.float32])
        return erf_vec(x).astype(np.float32)


def norm_cdf(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float32)
    return (0.5 * (1.0 + _erf(z / np.sqrt(2.0)))).astype(np.float32)


def clip_prob(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(x, eps, 1.0).astype(np.float32)


def safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return a / (b + eps)

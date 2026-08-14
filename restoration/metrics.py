"""
Pure measurement helpers shared by eval_dataset_a.py and eval_dataset_b.py.

No YOLO here -- precision/recall/mAP need the model loaded, so they live
in the eval scripts. This module only knows about pixels and time.
"""

import time
from functools import wraps

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(restored, clean):
    """
    Peak Signal-to-Noise Ratio in dB: how numerically close the restored
    image is to the clean ground truth. Higher is better. Only
    meaningful for Dataset A, where a clean reference exists.
    """
    return peak_signal_noise_ratio(clean, restored, data_range=255)


def ssim(restored, clean):
    """
    Structural Similarity: how similar the restored image's local
    structure (luminance/contrast/edges) is to the clean ground truth.
    Higher is better. Only meaningful for Dataset A.
    """
    return structural_similarity(clean, restored, channel_axis=2, data_range=255)


def sharpness(image):
    """
    Variance of the Laplacian: a cheap proxy for how in-focus/blurry a
    frame is. Higher = sharper. Used by sample_frames.py to bucket real
    video frames across the sharp -> heavily-blurred range, and could
    later drive an adaptive filter-selection threshold.
    """
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def timed(func):
    """
    Decorator that runs func normally but also returns elapsed wall time
    in milliseconds, so every filter's per-frame latency (section 12)
    is measured the same way without each eval script hand-rolling
    time.time() calls.

    Usage:
        result, elapsed_ms = timed(filters.bilateral_filter)(frame)
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return result, elapsed_ms

    return wrapper

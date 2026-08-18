"""
Synthetic degradation generators for Dataset A (controlled benchmark).

Used on clean images from data/clean/ for Dataset A. The lighting
functions here are also applied, as a deliberate exception, to two of
the seven real videos (video1, video2 -- see data/video_manifest.py)
because we have no real outdoor/bright footage and no time to reshoot;
that synthetic-on-real usage must stay clearly labeled as a synthetic
variant in eval_dataset_b.py, never conflated with the naturally dark
real footage in video5/video6. Noise and blur are never applied to any
real video -- that footage already has real motion blur/noise, so
degrading it further would just be fake data.

Each function returns a uint8 BGR image (same convention as cv2.imread)
plus the exact parameters used, so callers can log severity alongside
PSNR/SSIM/YOLO results in results.csv.
"""

import cv2
import numpy as np


def gaussian_noise(image, sigma):
    """
    Adds i.i.d. Gaussian sensor noise: I_noisy = I_clean + N(0, sigma^2).

    sigma is in pixel-intensity units (0-255 scale). Suggested severities:
        mild     sigma=10
        medium   sigma=25
        strong   sigma=45
    """
    noise = np.random.normal(0, sigma, image.shape)
    noisy = image.astype(np.float64) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def motion_blur(image, kernel_size, angle=0):
    """
    Convolves the image with a directional line kernel to simulate
    camera-motion blur (I_blurred = I_clean * h).

    kernel_size is the blur streak length in pixels. Suggested severities:
        mild     kernel_size=9
        medium   kernel_size=17
        severe   kernel_size=29
    angle is the sweep direction in degrees (0 = horizontal), matching
    the roughly horizontal sweeping motion of a cane.
    """
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    kernel[kernel_size // 2, :] = 1.0

    rot_matrix = cv2.getRotationMatrix2D(
        (kernel_size / 2, kernel_size / 2), angle, 1.0
    )
    kernel = cv2.warpAffine(kernel, rot_matrix, (kernel_size, kernel_size))
    kernel /= kernel.sum()

    return cv2.filter2D(image, -1, kernel)


def low_light(image, factor, noise_sigma):
    """
    Simulates dim/indoor lighting: darkens the image and adds
    proportionally more sensor noise, since real camera sensors raise
    gain (and therefore noise) to compensate in low light -- a flat
    brightness scale alone wouldn't be physically realistic.

    factor in (0, 1]: fraction of original brightness kept (lower = darker).
    noise_sigma: stddev of the additional low-light sensor noise.
    """
    darkened = image.astype(np.float64) * factor
    noise = np.random.normal(0, noise_sigma, image.shape)
    return np.clip(darkened + noise, 0, 255).astype(np.uint8)


def bright_glare(image, factor):
    """
    Simulates outdoor/backlit glare: pushes brightness up so highlights
    clip and wash out, the way direct sun or a bright window blows out
    part of a frame.

    factor > 1: brightness multiplier (higher = more washed out).
    """
    brightened = image.astype(np.float64) * factor
    return np.clip(brightened, 0, 255).astype(np.uint8)


# Named severity presets so eval_dataset_a.py can iterate a fixed,
# reproducible severity sweep instead of every script picking its own
# numbers.
NOISE_SEVERITIES = {
    "mild": {"sigma": 10},
    "medium": {"sigma": 25},
    "strong": {"sigma": 45},
}

BLUR_SEVERITIES = {
    "mild": {"kernel_size": 9, "angle": 0},
    "medium": {"kernel_size": 17, "angle": 0},
    "severe": {"kernel_size": 29, "angle": 0},
}

LOW_LIGHT_SEVERITIES = {
    "mild": {"factor": 0.6, "noise_sigma": 5},
    "medium": {"factor": 0.35, "noise_sigma": 12},
    "severe": {"factor": 0.15, "noise_sigma": 20},
}

GLARE_SEVERITIES = {
    "mild": {"factor": 1.3},
    "medium": {"factor": 1.6},
    "severe": {"factor": 2.0},
}

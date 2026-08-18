"""
Restoration / preprocessing methods under comparison.

Every filter shares the same signature: degraded_img (uint8 BGR) ->
restored_img (uint8 BGR). This lets eval_dataset_a.py / eval_dataset_b.py
swap methods interchangeably via the FILTERS registry below instead of
special-casing each one.
"""

import cv2
import numpy as np
from scipy.signal import wiener as scipy_wiener


def raw(image):
    """No-op baseline: the frame YOLO would see with zero preprocessing."""
    return image


def gaussian_filter(image, ksize=5, sigma=1.5):
    """Spatial-domain smoothing baseline. Blurs noise but also edges."""
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """
    Edge-preserving smoothing: averages nearby pixels only when they're
    also similar in intensity, so it denoises flat regions while mostly
    leaving object edges (what YOLO relies on) intact.
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def wiener_denoise(image, mysize=5):
    """
    Local adaptive Wiener filter for noise removal. Estimates local
    mean/variance in each mysize x mysize window and shrinks pixels
    toward that local mean more aggressively in low-variance (flat,
    noise-dominated) regions, less in high-variance (edge/texture)
    regions.
    """
    result = np.zeros_like(image, dtype=np.float64)
    for c in range(image.shape[2]):
        result[:, :, c] = scipy_wiener(
            image[:, :, c].astype(np.float64), mysize=mysize
        )
    return np.clip(result, 0, 255).astype(np.uint8)


def wiener_deconv(image, kernel_size=17, angle=0, K=0.01):
    """
    Frequency-domain Wiener deconvolution for motion blur.

    Models the degradation as G = H*F + N (convolution in the spatial
    domain == multiplication in the frequency domain), then inverts it
    per-frequency with the Wiener formula:

        F_hat = G * conj(H) / (|H|^2 + K)

    where K is a fixed noise-to-signal proxy that regularizes the
    division so near-zero frequencies in H don't blow up. Assumes a
    known/estimated blur kernel (kernel_size, angle) -- in a real
    deployment this would come from a blur-kernel estimator, but for
    Dataset A we know it exactly because we generated the blur with
    degrade.motion_blur() using the same parameters.
    """
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    kernel[kernel_size // 2, :] = 1.0
    rot_matrix = cv2.getRotationMatrix2D(
        (kernel_size / 2, kernel_size / 2), angle, 1.0
    )
    kernel = cv2.warpAffine(kernel, rot_matrix, (kernel_size, kernel_size))
    kernel /= kernel.sum()

    h, w = image.shape[:2]
    kernel_padded = np.zeros((h, w), dtype=np.float64)
    kh, kw = kernel.shape
    kernel_padded[:kh, :kw] = kernel
    kernel_padded = np.roll(kernel_padded, -(kh // 2), axis=0)
    kernel_padded = np.roll(kernel_padded, -(kw // 2), axis=1)

    H = np.fft.fft2(kernel_padded)
    H_conj = np.conj(H)
    denom = (H * H_conj) + K

    result = np.zeros_like(image, dtype=np.float64)
    for c in range(image.shape[2]):
        G = np.fft.fft2(image[:, :, c].astype(np.float64))
        F_hat = (H_conj / denom) * G
        result[:, :, c] = np.real(np.fft.ifft2(F_hat))

    return np.clip(result, 0, 255).astype(np.uint8)


def clahe_correction(image, clip_limit=2.0, tile_grid_size=8):
    """
    Contrast-Limited Adaptive Histogram Equalization: the standard
    lighting-correction baseline. Unlike the noise/blur filters above,
    this targets illumination, not corruption -- it redistributes the
    luminance histogram in local tiles to recover detail crushed into
    shadows (low light) or blown into highlights (glare), without
    touching color. Applied only to the L channel in LAB space so hue/
    saturation aren't distorted.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    l_corrected = clahe.apply(l_channel)

    corrected = cv2.merge((l_corrected, a_channel, b_channel))
    return cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)


# Central registry -- eval scripts iterate this dict so adding a new
# filter later means adding one line here, not touching the eval loop.
FILTERS = {
    "raw": raw,
    "gaussian": gaussian_filter,
    "bilateral": bilateral_filter,
    "wiener_denoise": wiener_denoise,
    "wiener_deconv": wiener_deconv,
    "clahe": clahe_correction,
}

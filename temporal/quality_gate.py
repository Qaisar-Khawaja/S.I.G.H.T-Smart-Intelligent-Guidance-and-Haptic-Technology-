"""
Part 4: lightweight sanity check on optical-flow alignment quality.

Cane-camera motion can be large enough that Farneback flow estimation
breaks down (textureless walls/floors give it nothing to lock onto;
large sweeps push content off the neighbor's edge entirely). Rather
than trusting every warped neighbor blindly, each one is scored right
after warping and either kept or rejected. Deliberately simple: three
independent threshold checks, no learned selector. A rejected neighbor
falls back to weight 0 in fusion.py (its share goes to the target
frame and/or the other neighbor).
"""

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.metrics import structural_similarity

# Loosely tuned by eyeballing warped output on a handful of
# moderate/severe_blur dev frames before running the full evaluation --
# not fit against the 84 evaluation frames themselves. See
# TEMPORAL_RESTORATION.md for how these behaved in practice.
MIN_VALID_FRACTION = 0.6    # fraction of the frame the neighbor actually covers post-warp
MAX_MEAN_ABS_DIFF = 40.0    # 0-255 scale, photometric diff on valid pixels
MIN_SSIM = 0.25             # structural similarity on valid pixels
MAX_MEAN_FLOW_MAG = 60.0    # pixels; catches flow estimation blowing up


@dataclass
class AlignmentReport:
    ok: bool
    valid_fraction: float
    mean_abs_diff: float
    ssim: float
    mean_flow_mag: float
    reason: str  # "ok", or the first check that failed


def assess_alignment(target_bgr, warped_bgr, valid_mask, flow):
    valid_fraction = float(valid_mask.mean())
    if valid_fraction < MIN_VALID_FRACTION:
        return AlignmentReport(False, valid_fraction, float("nan"), float("nan"), float("nan"), "low_valid_fraction")

    target_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    warped_gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    diff = np.abs(target_gray - warped_gray)
    mean_abs_diff = float(diff[valid_mask].mean())
    ssim_val = float(structural_similarity(target_gray, warped_gray, data_range=255))

    flow_mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    mean_flow_mag = float(flow_mag[valid_mask].mean())

    if mean_abs_diff > MAX_MEAN_ABS_DIFF:
        reason = "high_photometric_diff"
    elif ssim_val < MIN_SSIM:
        reason = "low_ssim"
    elif mean_flow_mag > MAX_MEAN_FLOW_MAG:
        reason = "excessive_flow_magnitude"
    else:
        reason = "ok"

    return AlignmentReport(reason == "ok", valid_fraction, mean_abs_diff, ssim_val, mean_flow_mag, reason)

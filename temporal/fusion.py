"""
Part 3: fuses the target frame with its aligned neighbor(s) into one
restored frame. Two variants:

  fixed_fusion            -- static weights (default 0.25 / 0.5 / 0.25)
  quality_weighted_fusion -- weights driven by each aligned frame's
                              sharpness, so a sharp neighbor
                              contributes more and a badly blurred one
                              contributes less

Both respect the alignment gate from quality_gate.py: a rejected or
missing neighbor gets weight 0, and its share is redistributed among
whatever's left (falling all the way back to the target frame if both
neighbors are unusable).
"""

import cv2
import numpy as np

DEFAULT_FIXED_WEIGHTS = {"prev": 0.25, "target": 0.5, "next": 0.25}
TARGET_MIN_WEIGHT = 0.3  # quality-weighted fusion never drops the target below this


def masked_sharpness(gray, mask=None):
    """Variance of Laplacian (see restoration.metrics.sharpness), restricted to `mask` if given."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    if mask is not None and mask.any():
        return float(lap[mask].var())
    return float(lap.var())


def _normalize(weights):
    total = sum(weights.values())
    if total <= 0:
        return {k: (1.0 if k == "target" else 0.0) for k in weights}
    return {k: v / total for k, v in weights.items()}


def _availability(prev_ok, next_ok):
    return {"prev": prev_ok, "target": True, "next": next_ok}


def _blend(target, warped_prev, warped_next, weights):
    fused = weights["target"] * target.astype(np.float32)
    if weights["prev"] > 0:
        fused += weights["prev"] * warped_prev.astype(np.float32)
    if weights["next"] > 0:
        fused += weights["next"] * warped_next.astype(np.float32)
    return np.clip(fused, 0, 255).astype(np.uint8)


def fixed_fusion(target, warped_prev, warped_next, prev_ok, next_ok, weights=None):
    """
    I_fused = w_prev * I'(t-1) + w_target * I(t) + w_next * I'(t+1).

    Rejected/missing neighbors (prev_ok / next_ok False) are zeroed out
    and the remaining weights renormalized to sum to 1, so the target
    frame always ends up with nonzero weight even if both neighbors
    are unusable.
    """
    weights = dict(weights or DEFAULT_FIXED_WEIGHTS)
    availability = _availability(prev_ok, next_ok)
    weights = _normalize({k: (w if availability[k] else 0.0) for k, w in weights.items()})
    return _blend(target, warped_prev, warped_next, weights), weights


def quality_weighted_fusion(target, warped_prev, warped_next, prev_ok, next_ok,
                             valid_prev=None, valid_next=None, target_min_weight=TARGET_MIN_WEIGHT):
    """
    Weighting formula:
      1. raw_i = sqrt(laplacian_variance(frame_i)) for each available
         frame i in {prev, target, next} (sqrt dampens the weight
         ratio when one frame is much sharper than another -- using
         raw variance directly produced unstably extreme weights
         during development, e.g. a sharp neighbor completely
         swamping a blurred target).
      2. w_i = raw_i / sum(raw)  (available frames only; unavailable
         ones are pinned to raw=0)
      3. if w_target < target_min_weight: pin w_target =
         target_min_weight and rescale the remaining
         (1 - target_min_weight) across whichever neighbor(s) are
         available, in proportion to their raw_i -- this keeps the
         annotated target frame from ever being outvoted entirely by
         a neighbor, since the target is the frame the ground truth
         boxes are drawn against.
    """
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    raw = {"target": masked_sharpness(target_gray) ** 0.5}

    availability = _availability(prev_ok, next_ok)
    raw["prev"] = masked_sharpness(cv2.cvtColor(warped_prev, cv2.COLOR_BGR2GRAY), valid_prev) ** 0.5 if availability["prev"] else 0.0
    raw["next"] = masked_sharpness(cv2.cvtColor(warped_next, cv2.COLOR_BGR2GRAY), valid_next) ** 0.5 if availability["next"] else 0.0

    weights = _normalize(raw)
    neighbor_total = weights["prev"] + weights["next"]

    if weights["target"] < target_min_weight:
        if neighbor_total > 0:
            remaining = 1.0 - target_min_weight
            weights = {
                "target": target_min_weight,
                "prev": remaining * (weights["prev"] / neighbor_total),
                "next": remaining * (weights["next"] / neighbor_total),
            }
        else:
            weights = {"target": 1.0, "prev": 0.0, "next": 0.0}

    return _blend(target, warped_prev, warped_next, weights), weights

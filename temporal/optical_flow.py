"""
Part 2: dense optical-flow alignment. Estimates per-pixel motion
between a target frame and a temporal neighbor, then warps the
neighbor into the target's coordinate system so it can be fused
without ghosting (the cane camera is moving, so naive averaging of
raw neighbor frames would double-expose anything that moved).

Uses OpenCV's Farneback dense flow -- no training, no external model,
cheap enough to run per-frame at evaluation time.
"""

import cv2
import numpy as np

# Farneback params: winsize=15 favors a smooth flow field over
# pixel-perfect displacement, which is the right tradeoff on footage
# that's already blurry/noisy (sharp per-pixel flow would mostly be
# fitting noise there).
FARNEBACK_PARAMS = dict(
    pyr_scale=0.5, levels=3, winsize=15, iterations=3,
    poly_n=5, poly_sigma=1.2, flags=0,
)


def compute_flow(target_gray, neighbor_gray):
    """
    Dense flow field flow[y, x] = (dx, dy) such that target(y, x)
    corresponds to neighbor(y + dy, x + dx). Direction matters here:
    this is flow FROM the target's pixel grid INTO the neighbor's, so
    it can be used directly as a remap source offset in warp_to_target.
    """
    return cv2.calcOpticalFlowFarneback(
        target_gray, neighbor_gray, None, **FARNEBACK_PARAMS
    )


def warp_to_target(neighbor_color, flow):
    """
    Resamples neighbor_color into the target frame's coordinate system:
    for each output pixel (x, y), samples neighbor_color at
    (x + flow_x, y + flow_y). This "pull" warp (as opposed to
    scattering neighbor pixels forward) never leaves holes in the
    output by construction.

    Returns (warped_bgr, valid_mask). valid_mask is False wherever the
    required source sample falls outside the neighbor frame -- content
    the neighbor simply never saw (e.g. something that entered frame
    only in the target, or large camera-sweep motion carrying a region
    off the neighbor's edge). Those pixels get a black fill from
    cv2.remap's border mode and must not be blended in as if they were
    real observations; quality_gate.py and fusion.py both key off this
    mask instead.
    """
    h, w = flow.shape[:2]
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x_coords + flow[..., 0]).astype(np.float32)
    map_y = (y_coords + flow[..., 1]).astype(np.float32)

    warped = cv2.remap(
        neighbor_color, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    valid_mask = (map_x >= 0) & (map_x <= w - 1) & (map_y >= 0) & (map_y <= h - 1)
    return warped, valid_mask


def align_neighbor(target_bgr, neighbor_bgr):
    """Full alignment step for one neighbor: gray conv, flow, warp."""
    target_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
    neighbor_gray = cv2.cvtColor(neighbor_bgr, cv2.COLOR_BGR2GRAY)
    flow = compute_flow(target_gray, neighbor_gray)
    warped, valid_mask = warp_to_target(neighbor_bgr, flow)
    return warped, valid_mask, flow

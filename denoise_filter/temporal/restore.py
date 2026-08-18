"""
Top-level entry point that ties Parts 1-4 together for one target
frame: take its NeighborBundle (temporal/neighbors.py), align each
neighbor with optical flow (temporal/optical_flow.py), gate the
alignment (temporal/quality_gate.py), then fuse (temporal/fusion.py).

align_neighbors() and fuse() are split apart deliberately: the two
fusion variants should see the *same* alignment so that a comparison
between them isolates the fusion strategy, not incidental differences
in flow estimation. restoration/eval_temporal.py aligns once per frame
and calls fuse() twice (fixed, quality_weighted).
"""

import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from temporal import fusion, optical_flow
from temporal.quality_gate import AlignmentReport, assess_alignment

# Maps the eval/report-facing method names (used in results CSVs, hero
# examples, plots) to the fuse() strategy name -- kept in one place so
# restoration/eval_temporal.py and analysis/temporal_hero_examples.py
# can't drift apart on this mapping.
EVAL_METHOD_TO_FUSE = {
    "temporal_fixed": "fixed",
    "temporal_quality_weighted": "quality_weighted",
}


@dataclass
class AlignedNeighbors:
    target: np.ndarray
    warped_prev: Optional[np.ndarray]
    valid_prev: Optional[np.ndarray]
    prev_report: Optional[AlignmentReport]
    warped_next: Optional[np.ndarray]
    valid_next: Optional[np.ndarray]
    next_report: Optional[AlignmentReport]
    offset: int

    @property
    def prev_ok(self):
        return self.prev_report is not None and self.prev_report.ok

    @property
    def next_ok(self):
        return self.next_report is not None and self.next_report.ok


@dataclass
class TemporalResult:
    restored: np.ndarray
    method: str
    weights: dict
    aligned: AlignedNeighbors

    def log_line(self, filename):
        w = {k: round(v, 3) for k, v in self.weights.items()}
        parts = [f"{filename} [{self.method}] offset={self.aligned.offset} weights={w}"]
        if self.aligned.prev_report is not None and not self.aligned.prev_ok:
            parts.append(f"prev REJECTED ({self.aligned.prev_report.reason})")
        elif self.aligned.prev_report is None:
            parts.append("prev MISSING (video boundary)")
        if self.aligned.next_report is not None and not self.aligned.next_ok:
            parts.append(f"next REJECTED ({self.aligned.next_report.reason})")
        elif self.aligned.next_report is None:
            parts.append("next MISSING (video boundary)")
        return " | ".join(parts)


def _align_and_gate(target, neighbor_frame):
    """Returns (warped, valid_mask, report), all None if no neighbor exists."""
    if neighbor_frame is None:
        return None, None, None
    if neighbor_frame.shape[:2] != target.shape[:2]:
        neighbor_frame = cv2.resize(neighbor_frame, (target.shape[1], target.shape[0]))
    warped, valid_mask, flow = optical_flow.align_neighbor(target, neighbor_frame)
    report = assess_alignment(target, warped, valid_mask, flow)
    return warped, valid_mask, report


def align_neighbors(target, bundle):
    """Aligns both neighbors in `bundle` against `target` (Parts 2 + 4)."""
    warped_prev, valid_prev, prev_report = _align_and_gate(target, bundle.prev_frame)
    warped_next, valid_next, next_report = _align_and_gate(target, bundle.next_frame)
    return AlignedNeighbors(target, warped_prev, valid_prev, prev_report,
                             warped_next, valid_next, next_report, bundle.offset)


def fuse(aligned, method, fixed_weights=None, target_min_weight=None):
    """Part 3: fuses already-aligned neighbors using `method` ('fixed' or 'quality_weighted')."""
    # Gated-out/missing neighbors fall back to the target's own pixels
    # so the fusion arithmetic can stay unconditional -- their weight
    # is 0 regardless (see fusion._availability), so this substitution
    # never actually contributes anything to the blend.
    warped_prev = aligned.warped_prev if aligned.warped_prev is not None else aligned.target
    warped_next = aligned.warped_next if aligned.warped_next is not None else aligned.target

    if method == "fixed":
        kwargs = {} if fixed_weights is None else {"weights": fixed_weights}
        restored, weights = fusion.fixed_fusion(
            aligned.target, warped_prev, warped_next, aligned.prev_ok, aligned.next_ok, **kwargs
        )
    elif method == "quality_weighted":
        kwargs = {} if target_min_weight is None else {"target_min_weight": target_min_weight}
        restored, weights = fusion.quality_weighted_fusion(
            aligned.target, warped_prev, warped_next, aligned.prev_ok, aligned.next_ok,
            aligned.valid_prev, aligned.valid_next, **kwargs
        )
    else:
        raise ValueError(f"unknown temporal method: {method}")

    return TemporalResult(restored, method, weights, aligned)


def save_debug_panel(bundle, aligned, results_by_method, out_path):
    """
    Part 2 deliverable: saves I(t-1) | I(t) | I(t+1) | warped I(t-1) |
    warped I(t+1) | fused (one column per configured method) as a
    single labeled strip, for visually sanity-checking alignment and
    fusion on a handful of frames.
    """
    h, w = aligned.target.shape[:2]
    blank = np.zeros((h, w, 3), dtype=np.uint8)

    def labeled(img, text):
        img = (img if img is not None else blank).copy()
        cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        return img

    panels = [
        labeled(bundle.prev_frame, "I(t-1) raw"),
        labeled(aligned.target, "I(t) target"),
        labeled(bundle.next_frame, "I(t+1) raw"),
        labeled(aligned.warped_prev, "warped I(t-1)" + ("" if aligned.prev_ok else " [rejected]")),
        labeled(aligned.warped_next, "warped I(t+1)" + ("" if aligned.next_ok else " [rejected]")),
    ]
    for method, result in results_by_method.items():
        panels.append(labeled(result.restored, f"fused ({method})"))

    panel = cv2.hconcat(panels)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, panel)

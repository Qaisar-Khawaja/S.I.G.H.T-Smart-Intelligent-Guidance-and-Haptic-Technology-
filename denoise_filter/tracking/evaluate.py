"""Evaluate continuous ByteTrack/BoT-SORT and finite detection persistence.

The seven source videos are tracked continuously using the raw YOLO11s
detections cached by analysis.temporal_oracle.  At the 84 annotated target
timestamps, tracker states are scored against the exact existing Dataset-B
ground truth.  Standard tracker output is compared with a small wrapper that
exposes unmatched-but-alive Kalman-predicted states for 1, 3, or 5 frames.

No pixels are restored or fused.  BoT-SORT's built-in sparse-optical-flow
global motion compensation is used only inside the tracker.

Usage:
    MPLCONFIGDIR=/private/tmp/sight-mpl-cache \
      venv/bin/python -m tracking.evaluate
    MPLCONFIGDIR=/private/tmp/sight-mpl-cache \
      venv/bin/python -m tracking.evaluate --force-outputs
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.byte_tracker import BYTETracker

from analysis.temporal_oracle import _nearest_frame, load_video_cache
from restoration.classes import CLASS_NAMES, RELEVANT_CLASSES
from restoration.detection_metrics import (
    IOU_THRESHOLD,
    extract_predictions,
    iou,
    match_detections,
    precision_recall_map_full,
)
from restoration.eval_temporal import load_dataset
from temporal.neighbors import load_metadata


RESULTS_CSV = Path("results/tracking_results.csv")
INSTANCES_CSV = Path("results/tracking_instance_analysis.csv")
RESCUE_SUMMARY_CSV = Path("results/tracking_rescue_summary.csv")
DETECTIONS_CSV = Path("results/tracking_detection_analysis.csv")
ORACLE_COMPARISON_CSV = Path("results/tracking_oracle_comparison.csv")
RUNTIME_CSV = Path("results/tracking_runtime.csv")
FRAME_STATES_JSONL = Path("results/tracking_frame_states.jsonl")
EXAMPLES_DIR = Path("results/tracking_examples")
PLOTS_DIR = Path("results/tracking_plots")
REPORT_MD = Path("TRACKING_PERSISTENCE.md")
DEFAULT_CACHE_DIR = Path("results/temporal_oracle_cache")
DEFAULT_CAUSAL_ORACLE_CSV = Path("results/temporal_oracle_causal_summary.csv")
MODEL_PATH = "yolo11s.pt"
CONFIDENCE = 0.15

TRACKER_CONFIGS = {
    "bytetrack": Path("tracking/bytetrack_sight.yaml"),
    "botsort": Path("tracking/botsort_sight.yaml"),
}
PERSISTENCE_WINDOWS = (1, 3, 5)
QUALITY_GROUPS = ("clear", "moderate_blur", "severe_blur")
CONFIDENCE_DECAY = 0.90


@dataclass
class TrackDetection:
    class_id: int
    confidence: float
    box: tuple[float, float, float, float]
    track_id: int | None = None
    missed_frames: int = 0
    persisted: bool = False


def _safe_output(path: Path, force: bool):
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --force-outputs")


def _write_csv(path: Path, rows, force=False):
    _safe_output(path, force)
    if not rows:
        raise RuntimeError(f"No rows available for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_tracker(name):
    with TRACKER_CONFIGS[name].open() as handle:
        args = SimpleNamespace(**yaml.safe_load(handle))
    args.device = "cpu"
    return BYTETracker(args) if name == "bytetrack" else BOTSORT(args)


def _boxes_for_tracker(detections, image_shape):
    data = np.asarray(
        [[*det.box, det.confidence, det.class_id] for det in detections],
        dtype=np.float32,
    ).reshape(-1, 6)
    return Boxes(data, image_shape[:2])


def _clip_box(box, width, height):
    x1, y1, x2, y2 = box
    clipped = (
        float(np.clip(x1, 0, width)),
        float(np.clip(y1, 0, height)),
        float(np.clip(x2, 0, width)),
        float(np.clip(y2, 0, height)),
    )
    return clipped if clipped[2] - clipped[0] >= 1 and clipped[3] - clipped[1] >= 1 else None


def _active_output(output, width, height):
    detections = []
    for row in output:
        class_id = int(row[6])
        if class_id not in RELEVANT_CLASSES:
            continue
        box = _clip_box(row[:4], width, height)
        if box is None:
            continue
        detections.append(TrackDetection(
            class_id=class_id,
            confidence=float(row[5]),
            box=box,
            track_id=int(row[4]),
        ))
    return detections


def _observed_output(output, source_detections, width, height):
    """Pass every current YOLO observation through, attaching IDs when confirmed.

    Ultralytics intentionally omits unconfirmed tracks from standard tracker
    output.  Suppressing a currently observed hazard is undesirable for this
    assistive pipeline, so persistence configurations keep the detector box
    and confidence for every current observation.  Confirmed tracker output's
    final column maps back to the source detection index.
    """
    track_id_by_detection_index = {int(row[7]): int(row[4]) for row in output}
    detections = []
    for index, det in enumerate(source_detections):
        box = _clip_box(det.box, width, height)
        if box is None:
            continue
        detections.append(TrackDetection(
            class_id=det.class_id,
            confidence=det.confidence,
            box=box,
            track_id=track_id_by_detection_index.get(index),
        ))
    return detections


def _persistent_output(tracker, observed, window, width, height):
    detections = list(observed)
    for track in tracker.lost_stracks:
        missed_frames = tracker.frame_id - track.end_frame
        if missed_frames < 1 or missed_frames > window:
            continue
        confidence = float(track.score) * (CONFIDENCE_DECAY ** missed_frames)
        if confidence < CONFIDENCE:
            continue
        class_id = int(track.cls)
        if class_id not in RELEVANT_CLASSES:
            continue
        box = _clip_box(track.xyxy, width, height)
        if box is None:
            continue
        # If the tracker failed to associate a current same-class detection,
        # do not output both that observation and a duplicate lost prediction.
        if any(det.class_id == class_id and iou(det.box, box) >= IOU_THRESHOLD for det in observed):
            continue
        detections.append(TrackDetection(
            class_id=class_id,
            confidence=confidence,
            box=box,
            track_id=int(track.track_id),
            missed_frames=missed_frames,
            persisted=True,
        ))
    return detections


def _serialize_detection(det):
    return {
        "class_id": det.class_id,
        "class_name": CLASS_NAMES.get(det.class_id, str(det.class_id)),
        "confidence": round(det.confidence, 6),
        "x1": round(det.box[0], 3),
        "y1": round(det.box[1], 3),
        "x2": round(det.box[2], 3),
        "y2": round(det.box[3], 3),
        "track_id": det.track_id,
        "missed_frames": det.missed_frames,
        "persisted": det.persisted,
    }


def run_trackers(predictions_by_video):
    """Run each tracker once; derive all persistence windows from its state."""
    frame_states = defaultdict(lambda: defaultdict(dict))
    runtime = {}
    total_expected = sum(len(frames) for frames in predictions_by_video.values())

    for tracker_name in TRACKER_CONFIGS:
        update_times, wrapper_times = [], []
        total_frames = 0
        print(f"Running {tracker_name} continuously over seven videos...")
        for video_num in range(1, 8):
            video_name = f"video{video_num}"
            tracker = _load_tracker(tracker_name)
            cap = cv2.VideoCapture(f"data/videos/{video_name}.mov")
            frame_idx = 0
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                height, width = image.shape[:2]
                boxes = _boxes_for_tracker(predictions_by_video[video_name].get(frame_idx, []), image.shape)
                start = time.perf_counter()
                output = tracker.update(boxes, img=image if tracker_name == "botsort" else None)
                update_times.append((time.perf_counter() - start) * 1000)

                active = _active_output(output, width, height)
                frame_states[tracker_name][video_name][frame_idx] = active
                wrapper_start = time.perf_counter()
                observed = _observed_output(
                    output, predictions_by_video[video_name].get(frame_idx, []), width, height
                )
                for window in PERSISTENCE_WINDOWS:
                    method = f"{tracker_name}_persist_{window}"
                    frame_states[method][video_name][frame_idx] = _persistent_output(
                        tracker, observed, window, width, height
                    )
                wrapper_times.append((time.perf_counter() - wrapper_start) * 1000)
                frame_idx += 1
            cap.release()
            total_frames += frame_idx
            print(f"  {video_name}: {frame_idx} frames")
        if total_frames != total_expected:
            raise RuntimeError(f"{tracker_name}: tracked {total_frames} frames, expected {total_expected}")
        runtime[tracker_name] = {
            "tracker_overhead_ms_mean": float(np.mean(update_times)),
            "tracker_overhead_ms_std": float(np.std(update_times)),
            "persistence_wrapper_ms_mean": float(np.mean(wrapper_times)),
            "num_frames": total_frames,
        }
    return frame_states, runtime


def infer_exact_targets(model, device, dataset, batch_size):
    all_detections = {}
    for start in range(0, len(dataset), batch_size):
        batch = dataset[start:start + batch_size]
        results = model([frame["image"] for frame in batch], verbose=False, device=device, conf=CONFIDENCE)
        for frame, result in zip(batch, results):
            detections = []
            for class_id, confidence, box in extract_predictions(result):
                detections.append(TrackDetection(class_id, confidence, tuple(box)))
            all_detections[frame["filename"]] = detections
    return all_detections


def match_frame(gts, detections):
    """Greedy Dataset-B matching with per-GT and per-detection details."""
    gt_entries = [{"class_id": c, "box": box, "det_idx": None} for c, box in gts]
    order = sorted(range(len(detections)), key=lambda idx: -detections[idx].confidence)
    detection_tp = [False] * len(detections)
    for det_idx in order:
        det = detections[det_idx]
        best_iou, best_gt = 0.0, None
        for gt_idx, gt in enumerate(gt_entries):
            if gt["det_idx"] is not None or gt["class_id"] != det.class_id:
                continue
            overlap = iou(gt["box"], det.box)
            if overlap > best_iou:
                best_iou, best_gt = overlap, gt_idx
        if best_gt is not None and best_iou >= IOU_THRESHOLD:
            gt_entries[best_gt]["det_idx"] = det_idx
            detection_tp[det_idx] = True
    gt_matches = [detections[entry["det_idx"]] if entry["det_idx"] is not None else None for entry in gt_entries]
    return gt_matches, detection_tp


def target_locations(dataset, metadata, timestamps_by_video):
    locations = {}
    for frame in dataset:
        meta = metadata[frame["filename"]]
        locations[frame["filename"]] = _nearest_frame(
            timestamps_by_video[frame["video_name"]], meta["timestamp_sec"]
        )
    return locations


def _exact_target_persistence(raw_detections, continuous_state):
    """Overlay only predicted lost states on the exact target-PNG baseline.

    Continuous tracking necessarily uses decoded video frames.  Dataset B's
    annotations and established raw baseline use separately saved exact PNGs;
    video 1 in particular is variable-frame-rate and can decode differently.
    At evaluation targets, retain the exact-PNG raw observations and add only
    genuine persisted states so decoder differences cannot masquerade as
    tracking rescues or losses.
    """
    output = list(raw_detections)
    for det in continuous_state:
        if not det.persisted:
            continue
        if any(raw.class_id == det.class_id and iou(raw.box, det.box) >= IOU_THRESHOLD for raw in raw_detections):
            continue
        output.append(det)
    return output


def evaluate(dataset, raw_by_filename, frame_states, locations):
    methods = ["raw"] + [
        method for tracker in TRACKER_CONFIGS
        for method in [tracker] + [f"{tracker}_persist_{window}" for window in PERSISTENCE_WINDOWS]
    ]
    pools = defaultdict(lambda: {"dets": [], "gts": []})
    instance_rows, detection_rows = [], []

    for frame in dataset:
        video, filename, stem = frame["video_name"], frame["filename"], frame["stem"]
        image_id = f"{video}/{stem}"
        base_idx = locations[filename]
        raw_detections = raw_by_filename[filename]
        raw_gt_matches, raw_detection_tp = match_frame(frame["gts"], raw_detections)

        detections_by_method = {"raw": raw_detections}
        for method in methods[1:]:
            continuous_state = frame_states[method][video][base_idx]
            detections_by_method[method] = (
                _exact_target_persistence(raw_detections, continuous_state)
                if "persist" in method else continuous_state
            )

        for method, detections in detections_by_method.items():
            for group in ("overall", frame["quality_group"]):
                pool = pools[(method, group)]
                pool["dets"].extend((image_id, det.class_id, det.confidence, det.box) for det in detections)
                pool["gts"].extend((image_id, class_id, box) for class_id, box in frame["gts"])

            gt_matches, detection_tp = match_frame(frame["gts"], detections)
            for det_idx, (det, is_tp) in enumerate(zip(detections, detection_tp)):
                detection_rows.append({
                    "method": method,
                    "frame_id": image_id,
                    "video": video,
                    "target_frame": base_idx,
                    "timestamp": frame["filename"].split("_t", 1)[1].split("_", 1)[0],
                    "quality_group": frame["quality_group"],
                    "detection_index": det_idx,
                    "class_id": det.class_id,
                    "class": CLASS_NAMES.get(det.class_id, str(det.class_id)),
                    "confidence": round(det.confidence, 4),
                    "track_id": det.track_id if det.track_id is not None else "",
                    "missed_frames_before_target": det.missed_frames,
                    "persisted": det.persisted,
                    "is_true_positive": is_tp,
                    "is_false_positive": not is_tp,
                    "is_stale_persisted_fp": det.persisted and not is_tp,
                    "x1": round(det.box[0], 3),
                    "y1": round(det.box[1], 3),
                    "x2": round(det.box[2], 3),
                    "y2": round(det.box[3], 3),
                })

            if method == "raw":
                continue
            for gt_idx, ((class_id, _), raw_match, tracked_match) in enumerate(
                zip(frame["gts"], raw_gt_matches, gt_matches)
            ):
                if raw_match is None and tracked_match is not None:
                    outcome = "rescued"
                elif raw_match is not None and tracked_match is None:
                    outcome = "lost"
                elif raw_match is not None and tracked_match is not None:
                    outcome = "both_detected"
                else:
                    outcome = "still_missed"
                instance_rows.append({
                    "method": method,
                    "gt_instance_id": f"{image_id}/gt{gt_idx:03d}",
                    "frame_id": image_id,
                    "gt_index": gt_idx,
                    "video": video,
                    "target_frame": base_idx,
                    "metadata_frame": frame["filename"].split("_f", 1)[1].split("_", 1)[0],
                    "timestamp": frame["filename"].split("_t", 1)[1].split("_", 1)[0],
                    "quality_group": frame["quality_group"],
                    "class_id": class_id,
                    "class": CLASS_NAMES.get(class_id, str(class_id)),
                    "raw_detected": raw_match is not None,
                    "tracked_detected": tracked_match is not None,
                    "raw_confidence": round(raw_match.confidence, 4) if raw_match else "",
                    "tracked_confidence": round(tracked_match.confidence, 4) if tracked_match else "",
                    "track_id": tracked_match.track_id if tracked_match else "",
                    "missed_frames_before_target": tracked_match.missed_frames if tracked_match else "",
                    "persisted_at_target": tracked_match.persisted if tracked_match else False,
                    "outcome": outcome,
                })

    result_rows = []
    for method in methods:
        for group in ("overall",) + QUALITY_GROUPS:
            pool = pools[(method, group)]
            precision, recall, map50, map50_95 = precision_recall_map_full(pool["dets"], pool["gts"])
            tp, fp, _ = match_detections(pool["dets"], pool["gts"])
            selected_detection_rows = [
                row for row in detection_rows
                if row["method"] == method and (group == "overall" or row["quality_group"] == group)
            ]
            selected_instance_rows = [
                row for row in instance_rows
                if row["method"] == method and (group == "overall" or row["quality_group"] == group)
            ]
            stale = [row for row in selected_detection_rows if row["is_stale_persisted_fp"]]
            result_rows.append({
                "method": method,
                "quality_group": group,
                "num_target_frames": len(dataset) if group == "overall" else sum(f["quality_group"] == group for f in dataset),
                "num_gt_instances": len(pool["gts"]),
                "true_positives": int(tp.sum()),
                "false_positives": int(fp.sum()),
                "false_negatives": len(pool["gts"]) - int(tp.sum()),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
                "map50": round(map50, 4),
                "map50_95": round(map50_95, 4),
                "rescued_vs_raw": sum(row["outcome"] == "rescued" for row in selected_instance_rows),
                "lost_vs_raw": sum(row["outcome"] == "lost" for row in selected_instance_rows),
                "both_detected": sum(row["outcome"] == "both_detected" for row in selected_instance_rows),
                "still_missed": sum(row["outcome"] == "still_missed" for row in selected_instance_rows),
                "persisted_detections": sum(bool(row["persisted"]) for row in selected_detection_rows),
                "stale_persisted_false_positives": len(stale),
                "mean_stale_duration_frames": round(float(np.mean([row["missed_frames_before_target"] for row in stale])), 3) if stale else 0.0,
                "max_stale_duration_frames": max((row["missed_frames_before_target"] for row in stale), default=0),
            })
    return result_rows, instance_rows, detection_rows


def rescue_summary(instance_rows):
    rows = []
    for method in sorted({row["method"] for row in instance_rows}):
        method_rows = [row for row in instance_rows if row["method"] == method]
        dimensions = {
            "severity": QUALITY_GROUPS,
            "class": sorted({row["class"] for row in method_rows}),
            "video": sorted({row["video"] for row in method_rows}),
        }
        keys = {"severity": "quality_group", "class": "class", "video": "video"}
        for dimension, values in dimensions.items():
            for value in values:
                selected = [row for row in method_rows if row[keys[dimension]] == value]
                counts = defaultdict(int)
                for row in selected:
                    counts[row["outcome"]] += 1
                rows.append({
                    "method": method,
                    "dimension": dimension,
                    "value": value,
                    "n_gt_instances": len(selected),
                    "rescued": counts["rescued"],
                    "lost": counts["lost"],
                    "both_detected": counts["both_detected"],
                    "still_missed": counts["still_missed"],
                })
    return rows


def benchmark_yolo(model, device, frames=120, warmup=10):
    cap = cv2.VideoCapture("data/videos/video6.mov")
    wall, preprocess, inference, postprocess = [], [], [], []
    for idx in range(frames + warmup):
        ok, frame = cap.read()
        if not ok:
            break
        start = time.perf_counter()
        result = model(frame, verbose=False, device=device, conf=CONFIDENCE)[0]
        elapsed = (time.perf_counter() - start) * 1000
        if idx >= warmup:
            wall.append(elapsed)
            preprocess.append(result.speed.get("preprocess", 0.0))
            inference.append(result.speed.get("inference", 0.0))
            postprocess.append(result.speed.get("postprocess", 0.0))
    cap.release()
    return {
        "benchmark_video": "video6",
        "benchmark_frames": len(wall),
        "yolo_preprocess_ms": float(np.mean(preprocess)),
        "yolo_inference_ms": float(np.mean(inference)),
        "yolo_postprocess_ms": float(np.mean(postprocess)),
        "yolo_wall_ms": float(np.mean(wall)),
        "yolo_wall_ms_std": float(np.std(wall)),
    }


def runtime_rows(result_rows, tracker_runtime, yolo_runtime):
    rows = []
    overall_methods = [row["method"] for row in result_rows if row["quality_group"] == "overall"]
    for method in overall_methods:
        tracker_name = next((name for name in TRACKER_CONFIGS if method.startswith(name)), None)
        tracker_ms = tracker_runtime[tracker_name]["tracker_overhead_ms_mean"] if tracker_name else 0.0
        wrapper_ms = tracker_runtime[tracker_name]["persistence_wrapper_ms_mean"] / 3 if "persist" in method else 0.0
        total = yolo_runtime["yolo_wall_ms"] + tracker_ms + wrapper_ms
        rows.append({
            "method": method,
            "device": "mps" if torch.backends.mps.is_available() else "cpu",
            "benchmark_video": yolo_runtime["benchmark_video"],
            "benchmark_frames": yolo_runtime["benchmark_frames"],
            "yolo_preprocess_ms": round(yolo_runtime["yolo_preprocess_ms"], 3),
            "yolo_inference_ms": round(yolo_runtime["yolo_inference_ms"], 3),
            "yolo_postprocess_ms": round(yolo_runtime["yolo_postprocess_ms"], 3),
            "yolo_wall_ms": round(yolo_runtime["yolo_wall_ms"], 3),
            "tracker_overhead_ms": round(tracker_ms, 3),
            "persistence_wrapper_ms": round(wrapper_ms, 3),
            "total_latency_ms": round(total, 3),
            "estimated_fps": round(1000 / total, 2),
        })
    return rows


def oracle_comparison(result_rows, causal_oracle_csv):
    with causal_oracle_csv.open() as handle:
        oracle_rows = {row["quality_group"]: row for row in csv.DictReader(handle)}
    result_lookup = {(row["method"], row["quality_group"]): row for row in result_rows}
    rows = []
    for tracker_name in TRACKER_CONFIGS:
        for window in PERSISTENCE_WINDOWS:
            method = f"{tracker_name}_persist_{window}"
            for group in ("overall",) + QUALITY_GROUPS:
                raw = float(oracle_rows[group]["raw_recall"])
                oracle = float(oracle_rows[group][f"past{window}_recall"])
                actual = float(result_lookup[(method, group)]["recall"])
                denominator = oracle - raw
                fraction = (actual - raw) / denominator if denominator > 0 else None
                rows.append({
                    "method": method,
                    "quality_group": group,
                    "window_frames": window,
                    "raw_recall": raw,
                    "past_only_oracle_recall": oracle,
                    "actual_tracker_recall": actual,
                    "available_oracle_headroom": round(denominator, 4),
                    "actual_recall_gain": round(actual - raw, 4),
                    "fraction_of_oracle_headroom_recovered": round(fraction, 4) if fraction is not None else "",
                })
    return rows


def write_frame_states(path, frame_states, timestamps_by_video, force=False):
    _safe_output(path, force)
    with path.open("w") as handle:
        for method in frame_states:
            for video in sorted(frame_states[method]):
                for frame_idx, detections in sorted(frame_states[method][video].items()):
                    handle.write(json.dumps({
                        "method": method,
                        "video": video,
                        "frame_index": frame_idx,
                        "timestamp": timestamps_by_video[video][frame_idx],
                        "detections": [_serialize_detection(det) for det in detections],
                    }, separators=(",", ":")) + "\n")


def _draw_detections(image, detections, color, prefix):
    for det in detections:
        x1, y1, x2, y2 = map(int, det.box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        track_id = getattr(det, "track_id", None)
        persisted = getattr(det, "persisted", False)
        missed_frames = getattr(det, "missed_frames", 0)
        track = f" id={track_id}" if track_id is not None else ""
        missed = f" m={missed_frames}" if persisted else ""
        cv2.putText(
            image,
            f"{prefix} {CLASS_NAMES.get(det.class_id, det.class_id)} {det.confidence:.2f}{track}{missed}",
            (x1, max(y1 - 4, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA,
        )


def _read_video_frames(video, indices):
    cap = cv2.VideoCapture(f"data/videos/{video}.mov")
    frames = {}
    idx = 0
    maximum = max(indices)
    while idx <= maximum:
        ok, image = cap.read()
        if not ok:
            break
        if idx in indices:
            frames[idx] = image
        idx += 1
    cap.release()
    return frames


def make_example_strip(frame, category, method, frame_states, raw_by_filename, raw_video, base_idx, focus_gt_idx=None):
    video = frame["video_name"]
    indices = [idx for idx in range(base_idx - 2, base_idx + 3) if idx >= 0]
    video_frames = _read_video_frames(video, indices)
    panels = []
    for idx in indices:
        image = video_frames[idx].copy()
        raw_detections = raw_video[video].get(idx, [])
        if idx == base_idx:
            image = frame["image"].copy()
            raw_detections = raw_by_filename[frame["filename"]]
            for gt_idx, (class_id, box) in enumerate(frame["gts"]):
                x1, y1, x2, y2 = map(int, box)
                color = (0, 0, 255) if focus_gt_idx is None or gt_idx == focus_gt_idx else (90, 90, 90)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        _draw_detections(image, raw_detections, (0, 210, 0), "RAW")
        tracking_detections = frame_states[method][video].get(idx, [])
        if idx == base_idx and "persist" in method:
            tracking_detections = _exact_target_persistence(raw_detections, tracking_detections)
        _draw_detections(image, tracking_detections, (255, 0, 255), "TRK")
        label = f"{idx - base_idx:+d}" if idx != base_idx else "TARGET"
        cv2.rectangle(image, (0, 0), (image.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(image, f"{label} frame={idx}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
        panels.append(cv2.resize(image, (320, 213)))
    strip = np.hstack(panels)
    out_dir = EXAMPLES_DIR / category
    out_dir.mkdir(parents=True, exist_ok=True)
    focus_suffix = f"_gt{focus_gt_idx:03d}" if focus_gt_idx is not None else ""
    out_path = out_dir / f"{video}_{frame['stem']}_{method}_{category}{focus_suffix}.jpg"
    cv2.imwrite(str(out_path), strip)
    return out_path


def generate_examples(dataset, method, instance_rows, detection_rows, frame_states, raw_by_filename, raw_video, locations):
    by_frame_id = {f"{frame['video_name']}/{frame['stem']}": frame for frame in dataset}
    outputs = defaultdict(list)
    method_instances = [row for row in instance_rows if row["method"] == method]
    lost_rows = sorted(
        [row for row in method_instances if row["outcome"] == "lost"],
        key=lambda row: (row["quality_group"] != "severe_blur", -float(row["raw_confidence"])),
    )
    if not lost_rows:
        base_tracker = method.split("_persist_", 1)[0]
        lost_rows = sorted(
            [row for row in instance_rows if row["method"] == base_tracker and row["outcome"] == "lost"],
            key=lambda row: (row["quality_group"] != "severe_blur", -float(row["raw_confidence"])),
        )
    category_rows = {
        "rescued": sorted(
            [row for row in method_instances if row["outcome"] == "rescued"],
            key=lambda row: (row["quality_group"] != "severe_blur", row["quality_group"] != "moderate_blur", -float(row["tracked_confidence"])),
        ),
        "lost": lost_rows,
        "neutral": [row for row in method_instances if row["outcome"] == "both_detected"],
    }
    for category, rows in category_rows.items():
        for row in rows[:2]:
            frame = by_frame_id[row["frame_id"]]
            example_method = row["method"]
            outputs[category].append(make_example_strip(
                frame, category, example_method, frame_states, raw_by_filename, raw_video,
                locations[frame["filename"]], focus_gt_idx=int(row["gt_index"]),
            ))

    stale = sorted(
        [row for row in detection_rows if row["method"] == method and row["is_stale_persisted_fp"]],
        key=lambda row: (-int(row["missed_frames_before_target"]), -float(row["confidence"])),
    )
    unique_stale = []
    seen_stale_frames = set()
    for row in stale:
        if row["frame_id"] in seen_stale_frames:
            continue
        seen_stale_frames.add(row["frame_id"])
        unique_stale.append(row)
        if len(unique_stale) == 2:
            break
    for row in unique_stale:
        frame = by_frame_id[row["frame_id"]]
        outputs["stale_false_positive"].append(make_example_strip(
            frame, "stale_false_positive", method, frame_states, raw_by_filename, raw_video,
            locations[frame["filename"]],
        ))
    return outputs


def generate_plots(result_rows, oracle_rows, runtime, best_method):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    lookup = {(row["method"], row["quality_group"]): row for row in result_rows}
    methods = ["raw", "bytetrack", "botsort", best_method]
    labels = ["Raw", "ByteTrack", "BoT-SORT", best_method.replace("_", " ")]
    x = np.arange(len(QUALITY_GROUPS))
    width = 0.19
    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, (method, label) in enumerate(zip(methods, labels)):
        ax.bar(x + (idx - 1.5) * width, [lookup[(method, g)]["recall"] for g in QUALITY_GROUPS], width, label=label)
    ax.set_xticks(x, ["Clear", "Moderate blur", "Severe blur"])
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 0.65)
    ax.legend(fontsize=8)
    ax.set_title("Detection recall by frame quality")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "tracking_recall_by_severity.png", dpi=180)
    plt.close(fig)

    overall = [row for row in result_rows if row["quality_group"] == "overall"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for row in overall:
        ax.scatter(row["recall"], row["precision"], s=55, label=row["method"])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–recall trade-off at confidence 0.15")
    ax.grid(alpha=0.25)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "tracking_precision_recall.png", dpi=180)
    plt.close(fig)

    persistence = [row for row in overall if "persist" in row["method"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(persistence))
    ax.bar(positions - 0.18, [row["rescued_vs_raw"] for row in persistence], 0.36, label="Rescued GT")
    ax.bar(positions + 0.18, [row["stale_persisted_false_positives"] for row in persistence], 0.36, label="Stale FP")
    ax.set_xticks(positions, [row["method"].replace("_persist_", " p") for row in persistence], rotation=25, ha="right")
    ax.set_ylabel("Instances")
    ax.set_title("Persistence benefit versus stale false positives")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "tracking_rescues_vs_stale_fp.png", dpi=180)
    plt.close(fig)

    best_tracker = "botsort" if best_method.startswith("botsort") else "bytetrack"
    compare = [row for row in oracle_rows if row["quality_group"] == "overall" and row["method"].startswith(best_tracker)]
    fig, ax = plt.subplots(figsize=(7, 5))
    windows = [row["window_frames"] for row in compare]
    ax.plot(windows, [row["past_only_oracle_recall"] for row in compare], marker="o", label="Past-only oracle")
    ax.plot(windows, [row["actual_tracker_recall"] for row in compare], marker="o", label="Actual tracker")
    ax.axhline(compare[0]["raw_recall"], color="black", linestyle="--", label="Raw")
    ax.set_xticks(windows)
    ax.set_xlabel("Persistence window (frames)")
    ax.set_ylabel("Overall recall")
    ax.set_title("Causal oracle ceiling versus actual tracking")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "tracking_vs_oracle.png", dpi=180)
    plt.close(fig)

    runtime_lookup = {row["method"]: row for row in runtime}
    runtime_methods = ["raw", "bytetrack", "botsort", best_method]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(runtime_methods)), [runtime_lookup[m]["estimated_fps"] for m in runtime_methods])
    ax.set_xticks(range(len(runtime_methods)), [m.replace("_", " ") for m in runtime_methods], rotation=20, ha="right")
    ax.set_ylabel("Estimated FPS")
    ax.set_title("Measured single-frame runtime")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "tracking_runtime_fps.png", dpi=180)
    plt.close(fig)


def write_report(
    result_rows,
    oracle_rows,
    runtime,
    best_method,
    examples,
    causal_oracle_csv,
    model_path,
    force=False,
):
    _safe_output(REPORT_MD, force)
    lookup = {(row["method"], row["quality_group"]): row for row in result_rows}
    runtime_lookup = {row["method"]: row for row in runtime}
    best = lookup[(best_method, "overall")]
    raw = lookup[("raw", "overall")]
    best_oracle = next(row for row in oracle_rows if row["method"] == best_method and row["quality_group"] == "overall")
    methods = ["raw", "bytetrack", "botsort", best_method]
    lines = [
        "# Detection-Level Tracking and Persistence",
        "",
        "## Research question",
        "",
        "Can causal object-state tracking bridge temporary raw-YOLO misses in cane-sweep video without creating unacceptable stale false positives? This follows the negative pixel-restoration result and the positive temporal-oracle headroom analysis.",
        "",
        "## Causal oracle sanity check",
        "",
        "| Group | Raw | Past 1 | Past 3 | Past 5 | Future 5 | ±5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    with causal_oracle_csv.open() as handle:
        causal = list(csv.DictReader(handle))
    for row in causal:
        lines.append(
            f"| {row['quality_group']} | {float(row['raw_recall']):.4f} | {float(row['past1_recall']):.4f} | "
            f"{float(row['past3_recall']):.4f} | {float(row['past5_recall']):.4f} | "
            f"{float(row['future5_recall']):.4f} | {float(row['bidirectional_pm5_recall']):.4f} |"
        )
    lines.extend([
        "",
        "Past-only headroom remains meaningful overall and for moderate blur. Severe blur has less causal headroom, and the bidirectional ceiling would require future-frame buffering.",
        "",
        "## Tracking method",
        "",
        f"ByteTrack and BoT-SORT are the implementations bundled with Ultralytics 8.4.120. Their default thresholds are pinned in `tracking/*_sight.yaml`: high/new-track 0.25, low 0.10, match 0.8, and lost-track buffer 30. YOLO input is `{model_path}` at confidence {CONFIDENCE}. BoT-SORT uses built-in sparse-optical-flow global motion compensation and no ReID.",
        "",
        f"Standard tracker output omits unconfirmed and unmatched tracks. For safety, persistence configurations pass every current raw YOLO observation through unchanged and add unmatched Kalman-predicted states for at most 1, 3, or 5 frames. At annotated targets, the pass-through observations come from the exact target PNG used by the established raw baseline; only genuine predicted states come from continuous video tracking, preventing variable-frame-rate decode differences from masquerading as rescues/losses. Predicted confidence = last detection confidence × {CONFIDENCE_DECAY}^missed_frames; boxes are clipped and decayed confidence below {CONFIDENCE} is dropped. A predicted lost box overlapping a current same-class observation at IoU >= {IOU_THRESHOLD} is suppressed as a duplicate.",
        "",
        "## Results",
        "",
        "| Method | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | TP | FP | Rescued | Lost | Stale persisted FP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for method in methods:
        row = lookup[(method, "overall")]
        lines.append(
            f"| {method} | {row['precision']:.4f} | {row['recall']:.4f} | {row['map50']:.4f} | "
            f"{row['map50_95']:.4f} | {row['true_positives']} | {row['false_positives']} | "
            f"{row['rescued_vs_raw']} | {row['lost_vs_raw']} | {row['stale_persisted_false_positives']} |"
        )
    lines.extend([
        "",
        "### Recall by severity",
        "",
        "| Method | Clear | Moderate blur | Severe blur | Overall |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for method in methods:
        lines.append(
            f"| {method} | {lookup[(method, 'clear')]['recall']:.4f} | "
            f"{lookup[(method, 'moderate_blur')]['recall']:.4f} | "
            f"{lookup[(method, 'severe_blur')]['recall']:.4f} | {lookup[(method, 'overall')]['recall']:.4f} |"
        )
    lines.extend([
        "",
        "### Persistence trade-off",
        "",
        "| Method | Rescued | Lost | Stale FP | Precision | Recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for tracker_name in TRACKER_CONFIGS:
        for window in PERSISTENCE_WINDOWS:
            method = f"{tracker_name}_persist_{window}"
            row = lookup[(method, "overall")]
            lines.append(
                f"| {method} | {row['rescued_vs_raw']} | {row['lost_vs_raw']} | "
                f"{row['stale_persisted_false_positives']} | {row['precision']:.4f} | {row['recall']:.4f} |"
            )
    lines.extend([
        "",
        f"The selected trade-off is `{best_method}`: configurations within 0.005 of the best overall F1 are tie-broken toward fewer stale persisted false positives, then higher precision.",
        "",
        "## Oracle exploitation",
        "",
        f"For {best_method}, actual overall recall is {best['recall']:.4f} versus a past-{best_oracle['window_frames']} oracle of {best_oracle['past_only_oracle_recall']:.4f}. It recovers {best_oracle['fraction_of_oracle_headroom_recovered']:.1%} of available causal recall headroom.",
        "",
        "| Group | Raw | Past oracle | Actual | Headroom recovered |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for group in ("overall",) + QUALITY_GROUPS:
        row = next(item for item in oracle_rows if item["method"] == best_method and item["quality_group"] == group)
        fraction = row["fraction_of_oracle_headroom_recovered"]
        fraction_text = "not applicable" if fraction == "" else f"{fraction:.1%}"
        lines.append(
            f"| {group} | {row['raw_recall']:.4f} | {row['past_only_oracle_recall']:.4f} | "
            f"{row['actual_tracker_recall']:.4f} | {fraction_text} |"
        )
    lines.extend([
        "",
        "## Runtime",
        "",
        "| Method | YOLO wall ms | Tracking ms | Total ms | FPS |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for method in methods:
        row = runtime_lookup[method]
        lines.append(
            f"| {method} | {row['yolo_wall_ms']:.3f} | {row['tracker_overhead_ms'] + row['persistence_wrapper_ms']:.3f} | "
            f"{row['total_latency_ms']:.3f} | {row['estimated_fps']:.2f} |"
        )
    lines.extend([
        "",
        (
            "The runtime benchmark is a sequential single-frame measurement on the current CPU environment. "
            + (
                "Earlier stored YOLO11s measurements were approximately 21.5 FPS for raw and 7.6–7.8 FPS for pixel-level temporal restoration; "
                if Path(model_path).name != "yolo11s.pt"
                else "Existing stored project measurements were approximately 21.5 FPS for raw and 7.6–7.8 FPS for pixel-level temporal restoration; "
            )
            + "hardware/runtime state can make direct absolute comparisons noisy."
        ),
        "",
        "## Pixel state versus object state",
        "",
        (
            "Farneback warping plus pixel fusion previously degraded severe-blur YOLO11s mAP@0.5 from 0.127 to 0.006/0.000. "
            if Path(model_path).name != "yolo11s.pt"
            else "Farneback warping plus pixel fusion degraded severe-blur mAP@0.5 from 0.127 to 0.006/0.000. "
        )
        + "Detection-level persistence leaves raw pixels untouched and instead carries a bounded, decaying tracker state. The current tracking result determines whether that distinction is practically useful rather than merely theoretically promising.",
        "",
        "## Examples and failure modes",
        "",
        f"Best successful strip: `{examples.get('rescued', ['none'])[0] if examples.get('rescued') else 'none'}`",
        "",
        f"Worst stale-failure strip: `{examples.get('stale_false_positive', ['none'])[0] if examples.get('stale_false_positive') else 'none'}`",
        "",
        "Green boxes are raw YOLO, magenta boxes are tracker/persistent state, and red target-frame boxes are GT. Lost and stale-false-positive folders are included alongside successful and neutral examples to avoid success-only selection.",
        f"Because the selected safety-preserving persistence wrapper loses no raw GT detections, its `lost/` examples show the standard {best_method.split('_persist_', 1)[0]} baseline suppressing current detections rather than a persistence loss.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "venv/bin/pip install -r requirements.txt",
        f"venv/bin/python -m analysis.temporal_oracle --model {model_path} --confidence {CONFIDENCE} --output-dir {RESULTS_CSV.parent} --report {RESULTS_CSV.parent / 'TEMPORAL_ORACLE.md'} --force-outputs",
        f"venv/bin/python -m analysis.temporal_oracle_causal --instances-csv {causal_oracle_csv.parent / 'temporal_oracle_instances.csv'} --output-csv {causal_oracle_csv} --force-output",
        f"MPLCONFIGDIR=/private/tmp/sight-mpl-cache venv/bin/python -m tracking.evaluate --model {model_path} --confidence {CONFIDENCE} --cache-dir {causal_oracle_csv.parent / 'temporal_oracle_cache'} --causal-oracle {causal_oracle_csv} --output-dir {RESULTS_CSV.parent} --report {REPORT_MD} --force-outputs",
        "```",
        "",
        "## Conclusion",
        "",
    ])
    recall_gain = best["recall"] - raw["recall"]
    if recall_gain >= 0.05 and best["precision"] >= 0.70:
        conclusion = "Detection-level temporal persistence provides a meaningful recall gain with tolerable precision, unlike pixel-level temporal fusion."
    elif recall_gain > 0 and best["precision"] >= raw["precision"] - 0.03:
        conclusion = "Detection persistence provides a modest positive result: it bridges a small number of short misses while essentially preserving raw precision, but remains far below the causal oracle under ego-motion."
    elif recall_gain > 0:
        conclusion = "Detection persistence provides a modest recall gain but pays a material precision cost and remains far below the causal oracle under ego-motion."
    else:
        conclusion = "The evaluated lightweight trackers do not exploit the temporal oracle headroom: association/prediction errors and stale states outweigh useful recovery under cane-camera ego-motion."
    lines.extend([conclusion, ""])
    REPORT_MD.write_text("\n".join(lines))


def print_summary(result_rows, best_method):
    print("\n=== TRACKING SUMMARY (overall) ===")
    for row in result_rows:
        if row["quality_group"] == "overall":
            print(
                f"{row['method']:24s} P={row['precision']:.3f} R={row['recall']:.3f} "
                f"mAP50={row['map50']:.3f} TP={row['true_positives']:3d} FP={row['false_positives']:3d} "
                f"rescue={row['rescued_vs_raw']:3d} lost={row['lost_vs_raw']:3d} staleFP={row['stale_persisted_false_positives']:3d}"
            )
    print(f"Best tracking method by overall F1: {best_method}")


def main():
    global CONFIDENCE, EXAMPLES_DIR, PLOTS_DIR, REPORT_MD, RESULTS_CSV

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--benchmark-frames", type=int, default=120)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--confidence", type=float, default=CONFIDENCE)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--causal-oracle",
        type=Path,
        default=DEFAULT_CAUSAL_ORACLE_CSV,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--report", type=Path, default=REPORT_MD)
    parser.add_argument("--expected-raw-tp", type=int)
    parser.add_argument("--expected-raw-fp", type=int)
    parser.add_argument("--force-outputs", action="store_true")
    args = parser.parse_args()

    CONFIDENCE = args.confidence
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = args.output_dir / "tracking_results.csv"
    RESULTS_CSV = results_csv
    instances_csv = args.output_dir / "tracking_instance_analysis.csv"
    rescue_summary_csv = args.output_dir / "tracking_rescue_summary.csv"
    detections_csv = args.output_dir / "tracking_detection_analysis.csv"
    oracle_comparison_csv = args.output_dir / "tracking_oracle_comparison.csv"
    runtime_csv = args.output_dir / "tracking_runtime.csv"
    frame_states_jsonl = args.output_dir / "tracking_frame_states.jsonl"
    EXAMPLES_DIR = args.output_dir / "tracking_examples"
    PLOTS_DIR = args.output_dir / "tracking_plots"
    REPORT_MD = args.report

    output_files = [results_csv, instances_csv, rescue_summary_csv, detections_csv,
                    oracle_comparison_csv, runtime_csv, frame_states_jsonl, REPORT_MD]
    for path in output_files:
        _safe_output(path, args.force_outputs)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for category in ("rescued", "lost", "neutral", "stale_false_positive"):
        (EXAMPLES_DIR / category).mkdir(parents=True, exist_ok=True)

    dataset = load_dataset()
    metadata = load_metadata()
    if len(dataset) != 84 or sum(len(frame["gts"]) for frame in dataset) != 344:
        raise RuntimeError("Dataset B no longer matches the expected 84 frames / 344 GT instances")

    predictions_by_video, timestamps_by_video = {}, {}
    for video_num in range(1, 8):
        video = f"video{video_num}"
        cache_path = args.cache_dir / f"{video}.jsonl"
        with cache_path.open() as handle:
            cache_metadata = json.loads(handle.readline())
        if Path(cache_metadata.get("model", "")).name != Path(args.model).name:
            raise RuntimeError(
                f"Cache/model mismatch for {cache_path}: "
                f"{cache_metadata.get('model')} != {args.model}"
            )
        if abs(float(cache_metadata.get("confidence_threshold", -1)) - CONFIDENCE) > 1e-9:
            raise RuntimeError(
                f"Cache confidence mismatch for {cache_path}: "
                f"{cache_metadata.get('confidence_threshold')} != {CONFIDENCE}"
            )
        predictions_by_video[video], timestamps_by_video[video] = load_video_cache(cache_path)

    locations = target_locations(dataset, metadata, timestamps_by_video)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(args.model)
    print(
        f"model={args.model}, device={device}, confidence={CONFIDENCE}; "
        "evaluating 84 exact target frames"
    )
    raw_by_filename = infer_exact_targets(model, device, dataset, args.batch_size)

    frame_states, tracker_runtime = run_trackers(predictions_by_video)
    result_rows, instance_rows, detection_rows = evaluate(
        dataset, raw_by_filename, frame_states, locations
    )
    raw = next(row for row in result_rows if row["method"] == "raw" and row["quality_group"] == "overall")
    expected_tp, expected_fp = args.expected_raw_tp, args.expected_raw_fp
    if expected_tp is None and expected_fp is None and Path(args.model).name == "yolo11s.pt":
        expected_tp, expected_fp = 90, 13
    if (expected_tp is None) != (expected_fp is None):
        raise ValueError("Pass both --expected-raw-tp and --expected-raw-fp, or neither")
    if expected_tp is not None and (raw["true_positives"], raw["false_positives"]) != (expected_tp, expected_fp):
        raise RuntimeError(
            f"Raw baseline mismatch: expected TP={expected_tp} FP={expected_fp}, "
            f"got TP={raw['true_positives']} FP={raw['false_positives']}"
        )

    tracking_overall = [row for row in result_rows if row["quality_group"] == "overall" and row["method"] != "raw"]
    persistence_overall = [row for row in tracking_overall if "persist" in row["method"]]
    best_f1 = max(row["f1"] for row in persistence_overall)
    near_best = [row for row in persistence_overall if row["f1"] >= best_f1 - 0.005]
    # Prefer the shortest/stalest-safe configuration when predictive quality
    # is effectively tied; this is an assistive system, not a recall-only race.
    best_method = min(
        near_best,
        key=lambda row: (row["stale_persisted_false_positives"], -row["precision"], -row["recall"]),
    )["method"]

    print("Benchmarking sequential single-frame YOLO runtime...")
    yolo_runtime = benchmark_yolo(model, device, frames=args.benchmark_frames)
    runtimes = runtime_rows(result_rows, tracker_runtime, yolo_runtime)
    oracle_rows = oracle_comparison(result_rows, args.causal_oracle)
    summaries = rescue_summary(instance_rows)

    for rows in (result_rows, instance_rows, detection_rows, oracle_rows, summaries, runtimes):
        for row in rows:
            row["model"] = Path(args.model).name
            row["confidence_threshold"] = CONFIDENCE

    _write_csv(results_csv, result_rows, args.force_outputs)
    _write_csv(instances_csv, instance_rows, args.force_outputs)
    _write_csv(rescue_summary_csv, summaries, args.force_outputs)
    _write_csv(detections_csv, detection_rows, args.force_outputs)
    _write_csv(oracle_comparison_csv, oracle_rows, args.force_outputs)
    _write_csv(runtime_csv, runtimes, args.force_outputs)
    write_frame_states(frame_states_jsonl, frame_states, timestamps_by_video, args.force_outputs)

    examples = generate_examples(
        dataset, best_method, instance_rows, detection_rows, frame_states,
        raw_by_filename, predictions_by_video, locations,
    )
    generate_plots(result_rows, oracle_rows, runtimes, best_method)
    write_report(
        result_rows,
        oracle_rows,
        runtimes,
        best_method,
        examples,
        args.causal_oracle,
        args.model,
        args.force_outputs,
    )
    print_summary(result_rows, best_method)
    print(f"Wrote {results_csv}, {instances_csv}, plots/examples, and {REPORT_MD}")


if __name__ == "__main__":
    main()

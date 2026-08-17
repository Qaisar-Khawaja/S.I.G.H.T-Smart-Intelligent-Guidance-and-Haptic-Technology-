"""Phase-1 temporal detection oracle for real cane-camera Dataset B.

The experiment deliberately operates on raw YOLO detections, not restored
pixels.  It first caches YOLO11s predictions for every decoded frame of the
seven source videos.  At each annotated target frame, normal Dataset-B
matching (same class, IoU >= 0.5) determines which GT instances raw YOLO
missed.  A conservative bidirectional short-tracklet association then asks
whether each miss has an unambiguous same-object detection within +/-1, 3,
or 5 decoded video frames.

Association is seeded by every target-frame GT box (not only the missed
ones), performed separately in each temporal direction, and is one-to-one
within a class.  Consecutive boxes are gated by normalized center motion,
area change, and aspect-ratio change.  Near-tied same-class assignments are
flagged as ambiguous and do not count as oracle recoveries.  This prevents,
for example, a detection of person B from automatically rescuing person A.

Outputs (all new; existing restoration results are untouched):
  results/temporal_oracle_predictions.jsonl
  results/temporal_oracle_summary.csv
  results/temporal_oracle_instances.csv
  TEMPORAL_ORACLE.md

Per-video prediction shards in results/temporal_oracle_cache/ make a rerun
resume-safe and avoid repeating full-video inference.

Usage:
    python -m analysis.temporal_oracle
    python -m analysis.temporal_oracle --rebuild-cache --force-outputs
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import torch
from ultralytics import YOLO

from restoration.classes import CLASS_NAMES, RELEVANT_CLASSES
from restoration.detection_metrics import IOU_THRESHOLD, extract_predictions, iou
from restoration.eval_temporal import CONFIDENCE, load_dataset
from temporal.neighbors import load_metadata, video_path_for


MODEL_PATH = "yolo11s.pt"
VIDEO_DIR = Path("data/videos")
CACHE_DIR = Path("results/temporal_oracle_cache")
PREDICTIONS_JSONL = Path("results/temporal_oracle_predictions.jsonl")
SUMMARY_CSV = Path("results/temporal_oracle_summary.csv")
INSTANCES_CSV = Path("results/temporal_oracle_instances.csv")
REPORT_MD = Path("TEMPORAL_ORACLE.md")

WINDOWS = (1, 3, 5)
QUALITY_GROUPS = ("clear", "moderate_blur", "severe_blur")

# Conservative association parameters.  Distances are normalized by the
# mean diagonal of the previous/predicted and candidate boxes, making the
# rule resolution- and object-scale-aware.  The gate relaxes only when a
# tracklet has had to coast through one or more missing frames.
CENTER_LIMIT_BASE = 0.70
CENTER_LIMIT_PER_MISSED_FRAME = 0.28
AREA_RATIO_BASE = 4.0
AREA_RATIO_PER_MISSED_FRAME = 1.25
ASPECT_RATIO_LIMIT = 3.0
MIN_ASSOCIATION_SCORE = 0.30
AMBIGUITY_MARGIN = 0.06


@dataclass
class Detection:
    class_id: int
    confidence: float
    box: tuple[float, float, float, float]


@dataclass
class TrackState:
    box: tuple[float, float, float, float]
    last_step: int = 0
    velocity: tuple[float, float, float, float] | None = None


def _xyxy_to_cxcywh(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, 1.0), max(y2 - y1, 1.0))


def _cxcywh_to_xyxy(values):
    cx, cy, width, height = values
    width, height = max(width, 1.0), max(height, 1.0)
    return (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)


def _predict_box(state: TrackState, step: int):
    if state.velocity is None:
        return state.box
    gap = step - state.last_step
    current = _xyxy_to_cxcywh(state.box)
    predicted = tuple(v + gap * dv for v, dv in zip(current, state.velocity))
    return _cxcywh_to_xyxy(predicted)


def association_score(reference_box, candidate_box, missed_frames: int):
    """Return a gated geometric similarity score, or None if implausible."""
    rcx, rcy, rw, rh = _xyxy_to_cxcywh(reference_box)
    ccx, ccy, cw, ch = _xyxy_to_cxcywh(candidate_box)
    ref_diag = math.hypot(rw, rh)
    cand_diag = math.hypot(cw, ch)
    scale = max((ref_diag + cand_diag) / 2, 20.0)
    center_norm = math.hypot(rcx - ccx, rcy - ccy) / scale

    ref_area, cand_area = rw * rh, cw * ch
    area_ratio = max(ref_area, cand_area) / max(min(ref_area, cand_area), 1.0)
    ref_aspect, cand_aspect = rw / rh, cw / ch
    aspect_ratio = max(ref_aspect, cand_aspect) / max(min(ref_aspect, cand_aspect), 1e-6)

    center_limit = CENTER_LIMIT_BASE + CENTER_LIMIT_PER_MISSED_FRAME * missed_frames
    area_limit = AREA_RATIO_BASE * (AREA_RATIO_PER_MISSED_FRAME ** missed_frames)
    overlap = iou(reference_box, candidate_box)
    if center_norm > center_limit or area_ratio > area_limit or aspect_ratio > ASPECT_RATIO_LIMIT:
        return None

    center_term = math.exp(-0.5 * (center_norm / 0.55) ** 2)
    size_term = math.exp(-abs(math.log(cand_area / ref_area)))
    aspect_term = math.exp(-abs(math.log(cand_aspect / ref_aspect)))
    score = 0.55 * center_term + 0.30 * overlap + 0.10 * size_term + 0.05 * aspect_term
    return score if score >= MIN_ASSOCIATION_SCORE else None


def _is_ambiguous(pair, feasible_pairs):
    gt_idx, det_idx, score = pair
    for other_gt, other_det, other_score in feasible_pairs:
        if (other_gt, other_det) == (gt_idx, det_idx):
            continue
        shares_endpoint = other_gt == gt_idx or other_det == det_idx
        if shares_endpoint and other_score >= score - AMBIGUITY_MARGIN:
            return True
    return False


def associate_direction(gts, frame_detections, base_idx: int, direction: int, max_window: int = 5):
    """Associate all GT identities into one temporal direction.

    Returns (matches_by_gt, ambiguous_steps_by_gt).  Each match entry is
    (absolute_frame_index, offset, Detection).  Ambiguous candidates do not
    update state and never count as a recovery.
    """
    states = {idx: TrackState(tuple(box)) for idx, (_, box) in enumerate(gts)}
    matches = defaultdict(list)
    ambiguous = defaultdict(list)

    for step in range(1, max_window + 1):
        frame_idx = base_idx + direction * step
        detections = frame_detections.get(frame_idx, [])
        by_class = defaultdict(list)
        for det_idx, det in enumerate(detections):
            by_class[det.class_id].append((det_idx, det))

        for class_id in {class_id for class_id, _ in gts}:
            gt_indices = [idx for idx, (gt_class, _) in enumerate(gts) if gt_class == class_id]
            class_dets = by_class.get(class_id, [])
            feasible = []
            for gt_idx in gt_indices:
                state = states[gt_idx]
                predicted = _predict_box(state, step)
                missed_frames = max(step - state.last_step - 1, 0)
                for local_det_idx, (_, det) in enumerate(class_dets):
                    score = association_score(predicted, det.box, missed_frames)
                    if score is not None:
                        feasible.append((gt_idx, local_det_idx, score))

            ambiguous_gt = set()
            for pair in feasible:
                if _is_ambiguous(pair, feasible):
                    ambiguous_gt.add(pair[0])
            for gt_idx in ambiguous_gt:
                ambiguous[gt_idx].append(direction * step)

            used_gts, used_dets = set(), set()
            for gt_idx, local_det_idx, score in sorted(feasible, key=lambda item: -item[2]):
                if gt_idx in used_gts or local_det_idx in used_dets:
                    continue
                if _is_ambiguous((gt_idx, local_det_idx, score), feasible):
                    continue
                _, det = class_dets[local_det_idx]
                state = states[gt_idx]
                old_values = _xyxy_to_cxcywh(state.box)
                new_values = _xyxy_to_cxcywh(det.box)
                elapsed = max(step - state.last_step, 1)
                state.velocity = tuple((new - old) / elapsed for old, new in zip(old_values, new_values))
                state.box = det.box
                state.last_step = step
                matches[gt_idx].append((frame_idx, direction * step, det))
                used_gts.add(gt_idx)
                used_dets.add(local_det_idx)

    return matches, ambiguous


def match_target_gts(gts, predictions):
    """Existing Dataset-B greedy target matching, with matched pred details."""
    gt_entries = [
        {"class_id": class_id, "box": tuple(box), "pred": None}
        for class_id, box in gts
    ]
    for det in sorted(predictions, key=lambda item: -item.confidence):
        best_iou, best_idx = 0.0, None
        for gt_idx, gt in enumerate(gt_entries):
            if gt["pred"] is not None or gt["class_id"] != det.class_id:
                continue
            overlap = iou(det.box, gt["box"])
            if overlap > best_iou:
                best_iou, best_idx = overlap, gt_idx
        if best_idx is not None and best_iou >= IOU_THRESHOLD:
            gt_entries[best_idx]["pred"] = det
    return [entry["pred"] for entry in gt_entries]


def _result_to_detections(result):
    return [Detection(class_id, confidence, tuple(box)) for class_id, confidence, box in extract_predictions(result)]


def _cache_metadata(video_name, video_path):
    stat = video_path.stat()
    return {
        "type": "metadata",
        "video_id": video_name,
        "source": str(video_path),
        "model": MODEL_PATH,
        "confidence_threshold": CONFIDENCE,
        "relevant_classes": sorted(RELEVANT_CLASSES),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
    }


def _cache_is_valid(path: Path, expected_metadata):
    if not path.exists():
        return False
    try:
        with path.open() as handle:
            actual = json.loads(handle.readline())
        return actual == expected_metadata
    except (OSError, json.JSONDecodeError):
        return False


def infer_video(model, device, video_name: str, video_path: Path, cache_path: Path, batch_size: int):
    expected_metadata = _cache_metadata(video_name, video_path)
    temp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source video: {video_path}")

    frame_records = []
    batch_frames, batch_info = [], []
    inference_ms = 0.0

    def flush_batch():
        nonlocal inference_ms
        if not batch_frames:
            return
        start = time.perf_counter()
        results = model(batch_frames, verbose=False, device=device, conf=CONFIDENCE)
        inference_ms += (time.perf_counter() - start) * 1000
        for (frame_idx, timestamp), result in zip(batch_info, results):
            detections = _result_to_detections(result)
            frame_records.append({
                "type": "frame",
                "video_id": video_name,
                "frame_index": frame_idx,
                "timestamp": round(timestamp, 6),
                "detections": [
                    {
                        "video_id": video_name,
                        "frame_index": frame_idx,
                        "timestamp": round(timestamp, 6),
                        "class_id": det.class_id,
                        "class_name": CLASS_NAMES.get(det.class_id, str(det.class_id)),
                        "confidence": round(det.confidence, 6),
                        "x1": round(det.box[0], 3),
                        "y1": round(det.box[1], 3),
                        "x2": round(det.box[2], 3),
                        "y2": round(det.box[3], 3),
                    }
                    for det in detections
                ],
            })
        batch_frames.clear()
        batch_info.clear()

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        batch_frames.append(frame)
        batch_info.append((frame_idx, timestamp))
        frame_idx += 1
        if len(batch_frames) >= batch_size:
            flush_batch()
    flush_batch()
    cap.release()

    with temp_path.open("w") as handle:
        handle.write(json.dumps(expected_metadata, separators=(",", ":")) + "\n")
        for record in frame_records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    os.replace(temp_path, cache_path)
    print(f"  {video_name}: cached {len(frame_records)} frames, inference {inference_ms / max(len(frame_records), 1):.1f} ms/frame")


def load_video_cache(path: Path):
    frames = {}
    timestamps = {}
    with path.open() as handle:
        next(handle)  # cache metadata
        for line in handle:
            record = json.loads(line)
            frame_idx = int(record["frame_index"])
            timestamps[frame_idx] = float(record["timestamp"])
            frames[frame_idx] = [
                Detection(
                    int(det["class_id"]),
                    float(det["confidence"]),
                    (float(det["x1"]), float(det["y1"]), float(det["x2"]), float(det["y2"])),
                )
                for det in record["detections"]
            ]
    return frames, timestamps


def consolidate_caches(cache_paths: Iterable[Path], output_path: Path):
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temp_path.open("w") as output:
        for cache_path in cache_paths:
            with cache_path.open() as source:
                next(source)  # omit per-shard metadata from the combined per-frame file
                for line in source:
                    output.write(line)
    os.replace(temp_path, output_path)


def _nearest_frame(timestamps, timestamp):
    return min(timestamps, key=lambda frame_idx: abs(timestamps[frame_idx] - timestamp))


def _round_or_blank(value, digits=4):
    return "" if value is None else round(value, digits)


def evaluate_oracle(model, device, dataset, metadata, predictions_by_video, timestamps_by_video, batch_size=8):
    print("Running exact target-frame baseline inference...")
    target_results = []
    for start in range(0, len(dataset), batch_size):
        batch = [frame["image"] for frame in dataset[start:start + batch_size]]
        target_results.extend(model(batch, verbose=False, device=device, conf=CONFIDENCE))
    target_predictions = [_result_to_detections(result) for result in target_results]

    rows = []
    for target_idx, (frame, target_preds) in enumerate(zip(dataset, target_predictions)):
        meta = metadata[frame["filename"]]
        video_name = frame["video_name"]
        video_predictions = predictions_by_video[video_name]
        video_timestamps = timestamps_by_video[video_name]
        base_idx = _nearest_frame(video_timestamps, meta["timestamp_sec"])
        target_matches = match_target_gts(frame["gts"], target_preds)

        prev_matches, prev_ambiguous = associate_direction(
            frame["gts"], video_predictions, base_idx, direction=-1, max_window=max(WINDOWS)
        )
        next_matches, next_ambiguous = associate_direction(
            frame["gts"], video_predictions, base_idx, direction=1, max_window=max(WINDOWS)
        )

        for gt_idx, ((class_id, box), raw_match) in enumerate(zip(frame["gts"], target_matches)):
            temporal_matches = prev_matches.get(gt_idx, []) + next_matches.get(gt_idx, [])
            temporal_matches.sort(key=lambda item: (abs(item[1]), item[1] > 0))
            prev = min(prev_matches.get(gt_idx, []), key=lambda item: abs(item[1]), default=None)
            nxt = min(next_matches.get(gt_idx, []), key=lambda item: abs(item[1]), default=None)
            ambiguous_offsets = sorted(set(prev_ambiguous.get(gt_idx, []) + next_ambiguous.get(gt_idx, [])), key=lambda x: (abs(x), x))

            recovered_distance = min((abs(item[1]) for item in temporal_matches), default=None)
            notes = []
            if raw_match is not None:
                notes.append("raw_detected")
            if ambiguous_offsets:
                notes.append("ambiguous_offsets=" + "|".join(f"{offset:+d}" for offset in ambiguous_offsets))
            if raw_match is None and recovered_distance is None:
                notes.append("no_unambiguous_nearby_detection_pm5")

            row = {
                "gt_instance_id": f"{video_name}/{frame['stem']}/gt{gt_idx:03d}",
                "video": video_name,
                "source_video": meta["source_video"],
                "target_frame": meta["frame_number"],
                "decoded_target_frame": base_idx,
                "timestamp": round(meta["timestamp_sec"], 6),
                "decoded_timestamp": round(video_timestamps[base_idx], 6),
                "target_alignment_delta_ms": round((video_timestamps[base_idx] - meta["timestamp_sec"]) * 1000, 3),
                "quality_group": frame["quality_group"],
                "class_id": class_id,
                "class": CLASS_NAMES.get(class_id, str(class_id)),
                "gt_x1": round(box[0], 3),
                "gt_y1": round(box[1], 3),
                "gt_x2": round(box[2], 3),
                "gt_y2": round(box[3], 3),
                "raw_detected": raw_match is not None,
                "raw_confidence": _round_or_blank(raw_match.confidence if raw_match else None),
                "nearest_prev_detection_frame": prev[0] if prev else "",
                "nearest_prev_detection_offset": prev[1] if prev else "",
                "nearest_prev_detection_timestamp": _round_or_blank(video_timestamps[prev[0]] if prev else None, 6),
                "nearest_prev_detection_confidence": _round_or_blank(prev[2].confidence if prev else None),
                "nearest_next_detection_frame": nxt[0] if nxt else "",
                "nearest_next_detection_offset": nxt[1] if nxt else "",
                "nearest_next_detection_timestamp": _round_or_blank(video_timestamps[nxt[0]] if nxt else None, 6),
                "nearest_next_detection_confidence": _round_or_blank(nxt[2].confidence if nxt else None),
                "nearest_unambiguous_detection_distance": recovered_distance if recovered_distance is not None else "",
                "recoverable_pm1": raw_match is None and recovered_distance is not None and recovered_distance <= 1,
                "recoverable_pm3": raw_match is None and recovered_distance is not None and recovered_distance <= 3,
                "recoverable_pm5": raw_match is None and recovered_distance is not None and recovered_distance <= 5,
                "ambiguous_pm1": raw_match is None and any(abs(offset) <= 1 for offset in ambiguous_offsets),
                "ambiguous_pm3": raw_match is None and any(abs(offset) <= 3 for offset in ambiguous_offsets),
                "ambiguous_pm5": raw_match is None and any(abs(offset) <= 5 for offset in ambiguous_offsets),
                "notes": ";".join(notes),
            }
            rows.append(row)

        if (target_idx + 1) % 12 == 0 or target_idx + 1 == len(dataset):
            print(f"  [{target_idx + 1}/{len(dataset)}] {frame['filename']}")
    return rows


def summarize(instance_rows, dataset):
    rows = []
    for group in ("overall",) + QUALITY_GROUPS:
        group_rows = instance_rows if group == "overall" else [row for row in instance_rows if row["quality_group"] == group]
        if not group_rows:
            continue
        raw_detected = sum(bool(row["raw_detected"]) for row in group_rows)
        raw_missed = len(group_rows) - raw_detected
        summary = {
            "quality_group": group,
            "n_target_frames": len(dataset) if group == "overall" else sum(frame["quality_group"] == group for frame in dataset),
            "n_gt_instances": len(group_rows),
            "raw_detected": raw_detected,
            "raw_missed": raw_missed,
            "raw_recall": round(raw_detected / len(group_rows), 4),
        }
        for window in WINDOWS:
            recoverable = sum(bool(row[f"recoverable_pm{window}"]) for row in group_rows)
            ambiguous = sum(bool(row[f"ambiguous_pm{window}"]) for row in group_rows)
            summary[f"raw_misses_recoverable_pm{window}"] = recoverable
            summary[f"temporal_oracle_pm{window}_detected"] = raw_detected + recoverable
            summary[f"temporal_oracle_pm{window}_recall"] = round((raw_detected + recoverable) / len(group_rows), 4)
            summary[f"ambiguous_misses_pm{window}"] = ambiguous
        summary["never_recovered_pm5"] = sum(
            not bool(row["raw_detected"]) and not bool(row["recoverable_pm5"])
            for row in group_rows
        )
        summary["ambiguous_only_pm5"] = sum(
            not bool(row["raw_detected"]) and not bool(row["recoverable_pm5"]) and bool(row["ambiguous_pm5"])
            for row in group_rows
        )
        rows.append(summary)
    return rows


def write_csv(path: Path, rows, force=False):
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing output {path}; pass --force-outputs to replace it")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary_rows, force=False):
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing report {path}; pass --force-outputs to replace it")
    by_group = {row["quality_group"]: row for row in summary_rows}
    labels = (("overall", "Overall"), ("clear", "Clear"), ("moderate_blur", "Moderate blur"), ("severe_blur", "Severe blur"))
    severe = by_group["severe_blur"]
    overall = by_group["overall"]

    lines = [
        "# Temporal Detection Oracle (Phase 1)",
        "",
        "## Motivation",
        "",
        f"Pixel-level restoration did not reliably improve Dataset B over raw {Path(MODEL_PATH).stem}, so this experiment tests whether correct object detections exist in nearby raw video frames and could theoretically bridge target-frame misses at the detection/state level.",
        "",
        "## Dataset and baseline",
        "",
        f"The evaluation uses all 84 annotated target frames ({overall['n_gt_instances']} scored GT instances) from the seven original cane-camera videos. Target-frame detections use the exact annotated PNGs. `{MODEL_PATH}`, confidence {CONFIDENCE}, the repository's relevant-class filter, and same-class IoU >= {IOU_THRESHOLD} matching are unchanged from Dataset B.",
        "The ±1, ±3, and ±5 windows correspond to approximately ±33 ms, ±100 ms, and ±167 ms at 30 FPS (video 1 metadata reports 29.7 FPS; the others report 30 FPS).",
        "",
        "## Association method",
        "",
        "Each target-frame GT box seeds independent forward and backward tracklets through the cached full-video detections. Association is same-class and one-to-one, with gates on normalized center displacement, area ratio, and aspect-ratio change. A constant-velocity estimate is used only after a first association; otherwise the GT/last box is held during short gaps. Near-tied assignments (score margin <= 0.06) are flagged as ambiguous, do not update a tracklet, and are not credited as recoveries. This is intentionally conservative where several people or chairs are present.",
        "",
        "## Results",
        "",
        "| Group | GT | Raw recall | Oracle ±1 | Oracle ±3 | Oracle ±5 | Raw misses | Recoverable ±1 | Recoverable ±3 | Recoverable ±5 | Never recovered ±5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, label in labels:
        row = by_group[key]
        lines.append(
            f"| {label} | {row['n_gt_instances']} | {row['raw_recall']:.4f} | "
            f"{row['temporal_oracle_pm1_recall']:.4f} | {row['temporal_oracle_pm3_recall']:.4f} | "
            f"{row['temporal_oracle_pm5_recall']:.4f} | {row['raw_missed']} | "
            f"{row['raw_misses_recoverable_pm1']} | {row['raw_misses_recoverable_pm3']} | "
            f"{row['raw_misses_recoverable_pm5']} | {row['never_recovered_pm5']} |"
        )

    meaningful = (
        severe["temporal_oracle_pm5_recall"] - severe["raw_recall"] >= 0.10
        or overall["temporal_oracle_pm5_recall"] - overall["raw_recall"] >= 0.05
    )
    decision = (
        "The conservative oracle shows meaningful short-window headroom, so Phase 2 tracking/persistence is justified."
        if meaningful
        else "The conservative oracle shows little short-window headroom, so a complex Phase 2 tracker is not justified by these data."
    )
    lines.extend([
        "",
        "## Severe-blur focus",
        "",
        f"Severe blur contains {severe['n_gt_instances']} GT instances: {severe['raw_detected']} raw detections and {severe['raw_missed']} raw misses. Of those misses, {severe['raw_misses_recoverable_pm1']} are recoverable within ±1, {severe['raw_misses_recoverable_pm3']} within ±3, and {severe['raw_misses_recoverable_pm5']} within ±5. {severe['never_recovered_pm5']} remain unrecovered within ±5; {severe['ambiguous_only_pm5']} of those have only ambiguous nearby candidates.",
        "",
        "## Phase 2 decision",
        "",
        decision,
        "",
        "This oracle is an upper-bound diagnostic, not a deployable tracker: it uses the target-frame GT box to seed identity and deliberately excludes ambiguous cases. A real tracker can recover only a subset and must also be judged on precision and stale-track false positives.",
        "",
    ])
    path.write_text("\n".join(lines))
    return meaningful


def print_summary(summary_rows):
    print("\n=== TEMPORAL ORACLE SUMMARY ===")
    print("group             GT   raw     pm1     pm3     pm5   rescued(1/3/5)  never5")
    for row in summary_rows:
        print(
            f"{row['quality_group']:16s} {row['n_gt_instances']:>3d}  "
            f"{row['raw_recall']:.3f}   {row['temporal_oracle_pm1_recall']:.3f}   "
            f"{row['temporal_oracle_pm3_recall']:.3f}   {row['temporal_oracle_pm5_recall']:.3f}   "
            f"{row['raw_misses_recoverable_pm1']:>3d}/{row['raw_misses_recoverable_pm3']:>3d}/{row['raw_misses_recoverable_pm5']:>3d}       "
            f"{row['never_recovered_pm5']:>3d}"
        )


def main():
    global MODEL_PATH, CONFIDENCE

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--confidence", type=float, default=CONFIDENCE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for cache shards and oracle CSV/JSONL outputs",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPORT_MD,
        help="Markdown report path",
    )
    parser.add_argument(
        "--expected-raw-recall",
        type=float,
        help="Optional target-frame baseline check",
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--force-outputs", action="store_true")
    args = parser.parse_args()

    MODEL_PATH = args.model
    CONFIDENCE = args.confidence
    cache_dir = args.output_dir / "temporal_oracle_cache"
    predictions_jsonl = args.output_dir / "temporal_oracle_predictions.jsonl"
    summary_csv = args.output_dir / "temporal_oracle_summary.csv"
    instances_csv = args.output_dir / "temporal_oracle_instances.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset()
    metadata = load_metadata()
    if len(dataset) != 84:
        raise RuntimeError(f"Expected 84 Dataset-B target frames, found {len(dataset)}")
    if sum(len(frame["gts"]) for frame in dataset) != 344:
        raise RuntimeError("Dataset-B GT count changed; expected 344 scored instances")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(MODEL_PATH)
    print(
        f"Temporal oracle: model={MODEL_PATH}, device={device}, "
        f"confidence={CONFIDENCE}, target IoU={IOU_THRESHOLD}"
    )

    cache_paths = []
    for video_num in range(1, 8):
        video_name = f"video{video_num}"
        video_path = VIDEO_DIR / f"{video_name}.mov"
        cache_path = cache_dir / f"{video_name}.jsonl"
        expected = _cache_metadata(video_name, video_path)
        if args.rebuild_cache or not _cache_is_valid(cache_path, expected):
            infer_video(model, device, video_name, video_path, cache_path, args.batch_size)
        else:
            print(f"  {video_name}: reusing valid cache {cache_path}")
        cache_paths.append(cache_path)

    if predictions_jsonl.exists() and not args.force_outputs:
        raise FileExistsError(f"Refusing to overwrite {predictions_jsonl}; pass --force-outputs")
    consolidate_caches(cache_paths, predictions_jsonl)
    print(f"Wrote reusable full-video predictions to {predictions_jsonl}")

    predictions_by_video, timestamps_by_video = {}, {}
    for video_num, cache_path in enumerate(cache_paths, start=1):
        predictions_by_video[f"video{video_num}"], timestamps_by_video[f"video{video_num}"] = load_video_cache(cache_path)

    instance_rows = evaluate_oracle(
        model, device, dataset, metadata, predictions_by_video, timestamps_by_video,
        batch_size=args.batch_size,
    )
    raw_recall = sum(bool(row["raw_detected"]) for row in instance_rows) / len(instance_rows)
    expected_raw_recall = args.expected_raw_recall
    if expected_raw_recall is None and Path(MODEL_PATH).name == "yolo11s.pt":
        expected_raw_recall = 0.2616
    if expected_raw_recall is not None and abs(raw_recall - expected_raw_recall) > 0.005:
        raise RuntimeError(
            "Raw baseline validation failed: expected approximately "
            f"{expected_raw_recall:.4f}, got {raw_recall:.4f}"
        )

    summary_rows = summarize(instance_rows, dataset)
    write_csv(instances_csv, instance_rows, force=args.force_outputs)
    write_csv(summary_csv, summary_rows, force=args.force_outputs)
    meaningful = write_report(args.report, summary_rows, force=args.force_outputs)
    print_summary(summary_rows)
    print(f"\nWrote {len(instance_rows)} instances to {instances_csv}")
    print(f"Wrote summary to {summary_csv}")
    print(f"Wrote report to {args.report}")
    print("Phase 2 decision:", "PROCEED" if meaningful else "STOP")


if __name__ == "__main__":
    main()

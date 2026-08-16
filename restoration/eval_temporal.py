"""
Part 5-9 orchestrator: evaluates whether temporal (multi-frame,
optical-flow-aligned) restoration improves YOLO detection on Dataset B,
compared with raw input and the existing single-frame filters.

Uses the same 84 ground-truth-annotated real frames, YOLO11s model,
confidence threshold (0.15, matching main.py's live setting -- see
eval_dataset_b.py's module docstring), and IoU/mAP scoring
(restoration.detection_metrics) as eval_dataset_b.py. Only the set of
preprocessing methods under comparison is extended.

Methods evaluated:
  raw                        restoration.filters.raw (no-op baseline)
  wiener_denoise              restoration.filters.wiener_denoise
  wiener_deconv                restoration.filters.wiener_deconv (default
                               kernel -- real footage has no known blur
                               kernel, same caveat as eval_dataset_b.py)
  clahe                       restoration.filters.clahe_correction
  temporal_fixed               temporal.restore, fixed 0.25/0.5/0.25 fusion
  temporal_quality_weighted     temporal.restore, sharpness-weighted fusion

Neighbor frames come from temporal.neighbors (decoded fresh from
data/videos/videoN.mov); the target frame itself is always the exact
promoted image in data/frames_real/, so ground truth in
data/real_labels/ stays valid for every method, including the temporal
ones.

Outputs (never overwrites eval_dataset_b.py's results/results_dataset_b_*.csv):
  results/temporal_results.csv         Part 9 -- one row per method x
                                        {overall, clear, moderate_blur,
                                        severe_blur}: precision/recall/
                                        map50/map50_95 + timing.
  results/temporal_frame_analysis.csv  Part 7 -- one row per frame x
                                        ground-truth class x temporal
                                        method, comparing raw vs that
                                        method's detections/confidence.
  results/temporal_examples/debug/*.jpg  Part 2 -- alignment/fusion
                                        debug panels for a handful of
                                        frames spanning all three
                                        quality groups.

Usage:
    python -m restoration.eval_temporal
    python -m restoration.eval_temporal --offset 2 --debug-per-group 3
"""

import argparse
import csv
import os
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from restoration import filters
from restoration.detection_metrics import (
    extract_predictions,
    iou,
    load_ground_truth,
    match_detections,
    precision_recall_map_full,
)
from temporal import neighbors as neighbors_mod
from temporal import restore

FRAMES_DIR = "data/frames_real"
LABELS_DIR = "data/real_labels"
METADATA_CSV = "data/frames_real_metadata.csv"

RESULTS_CSV = "results/temporal_results.csv"
FRAME_ANALYSIS_CSV = "results/temporal_frame_analysis.csv"
DEBUG_DIR = "results/temporal_examples/debug"

SINGLE_FRAME_METHODS = ["raw", "wiener_denoise", "wiener_deconv", "clahe"]
TEMPORAL_METHODS = ["temporal_fixed", "temporal_quality_weighted"]
ALL_METHODS = SINGLE_FRAME_METHODS + TEMPORAL_METHODS

QUALITY_GROUPS = ["clear", "moderate_blur", "severe_blur"]

# Same confidence threshold main.py's live decision pipeline uses (see
# eval_dataset_b.py) -- keeps this evaluation comparable to that one
# and to how the cane would actually behave.
CONFIDENCE = 0.15

# A confidence delta below this is treated as "no meaningful change"
# for the Part 7 rescue/lost/confidence-shift bookkeeping.
CONFIDENCE_DELTA = 0.05


def load_dataset():
    """Returns list of dicts: filename, video_name, quality_group, image, gts (class_id, box)."""
    meta = neighbors_mod.load_metadata(METADATA_CSV)
    dataset = []
    for filename, row in sorted(meta.items()):
        video_num = int(row["video_id"].split("_")[0][1:])
        video_name = f"video{video_num}"
        img_path = os.path.join(FRAMES_DIR, video_name, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"  WARNING: missing target frame {img_path}, skipping")
            continue

        stem = os.path.splitext(filename)[0]
        label_path = os.path.join(LABELS_DIR, video_name, stem + ".txt")
        gts = load_ground_truth(label_path, img.shape[1], img.shape[0])

        dataset.append({
            "filename": filename,
            "stem": stem,
            "video_name": video_name,
            "quality_group": row["quality_group"],
            "image": img,
            "gts": gts,
        })
    return dataset


def run_yolo(model, device, image):
    start = time.perf_counter()
    results = model(image, verbose=False, device=device, conf=CONFIDENCE)
    yolo_ms = (time.perf_counter() - start) * 1000
    return extract_predictions(results[0]), yolo_ms


def frame_class_comparison(gts, raw_preds, method_preds):
    """
    Part 7: for every ground-truth class present in this frame, compares
    how many instances raw vs `method` correctly detected (IoU>=0.5,
    matching class), and the best confidence among the matches.

    Bookkeeping is per (frame, class) rather than per individual GT box
    -- when a frame has multiple GT boxes of the same class, this counts
    how many of them each method caught rather than trying to identify
    which specific instance was "rescued", which would require object
    identity that doesn't exist here. Reuses detection_metrics'
    tested greedy IoU matcher scoped to a single synthetic image id, so
    the semantics match precision_recall_map exactly.
    """
    classes_present = {c for c, _ in gts}
    rows = []
    for class_id in classes_present:
        class_gts = [("img", class_id, box) for c, box in gts if c == class_id]
        n_gt = len(class_gts)

        raw_dets = [("img", c, conf, box) for c, conf, box in raw_preds if c == class_id]
        method_dets = [("img", c, conf, box) for c, conf, box in method_preds if c == class_id]

        raw_tp, _, raw_sorted = match_detections(raw_dets, class_gts)
        method_tp, _, method_sorted = match_detections(method_dets, class_gts)

        raw_tp_count = int(raw_tp.sum())
        method_tp_count = int(method_tp.sum())

        raw_conf = max((d[2] for d, is_tp in zip(raw_sorted, raw_tp) if is_tp), default=None)
        method_conf = max((d[2] for d, is_tp in zip(method_sorted, method_tp) if is_tp), default=None)

        if method_tp_count > raw_tp_count:
            outcome = "rescued"
        elif method_tp_count < raw_tp_count:
            outcome = "lost"
        elif raw_tp_count == 0:
            outcome = "still_missed"
        elif method_conf is not None and raw_conf is not None and method_conf - raw_conf > CONFIDENCE_DELTA:
            outcome = "confidence_improved"
        elif method_conf is not None and raw_conf is not None and raw_conf - method_conf > CONFIDENCE_DELTA:
            outcome = "confidence_degraded"
        else:
            outcome = "no_change"

        rows.append({
            "class_id": class_id,
            "n_gt": n_gt,
            "raw_detected": raw_tp_count > 0,
            "raw_tp_count": raw_tp_count,
            "raw_confidence": round(raw_conf, 4) if raw_conf is not None else "",
            "method_detected": method_tp_count > 0,
            "method_tp_count": method_tp_count,
            "method_confidence": round(method_conf, 4) if method_conf is not None else "",
            "outcome": outcome,
        })
    return rows


def compute_metrics_rows(pools, timing):
    """pools: {(method, group): {"dets":[], "gts":[]}}; timing: {(method, group): {"preprocess_ms":[], "total_ms":[]}}."""
    rows = []
    for method in ALL_METHODS:
        for group in ["overall"] + QUALITY_GROUPS:
            key = (method, group)
            pool = pools.get(key, {"dets": [], "gts": []})
            times = timing.get(key, {"preprocess_ms": [], "total_ms": []})
            if not pool["gts"]:
                continue
            precision, recall, map50, map50_95 = precision_recall_map_full(pool["dets"], pool["gts"])
            preprocess = np.array(times["preprocess_ms"]) if times["preprocess_ms"] else np.array([0.0])
            total = np.array(times["total_ms"]) if times["total_ms"] else np.array([0.0])
            rows.append({
                "method": method,
                "quality_group": group,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "map50": round(map50, 4),
                "map50_95": round(map50_95, 4),
                "preprocess_ms_mean": round(float(preprocess.mean()), 3),
                "preprocess_ms_std": round(float(preprocess.std()), 3),
                "total_latency_ms": round(float(total.mean()), 3),
                "estimated_fps": round(1000.0 / float(total.mean()), 2) if total.mean() > 0 else 0.0,
                "num_frames": len({g[0] for g in pool["gts"]}) if pool["gts"] else 0,
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offset", type=int, default=1,
                         help="neighbor frame offset, i.e. use I(t-offset)/I(t+offset) (default: 1)")
    parser.add_argument("--debug-per-group", type=int, default=2,
                         help="number of debug alignment panels to save per quality group (default: 2)")
    parser.add_argument("--limit", type=int, default=None,
                         help="only process the first N frames (debugging/smoke-testing)")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)

    print("Loading Dataset B (84 annotated real frames)...")
    dataset = load_dataset()
    if args.limit:
        dataset = dataset[:args.limit]
    print(f"  {len(dataset)} frames loaded")

    print(f"Building temporal neighbor bundles (offset={args.offset})...")
    bundles = neighbors_mod.build_neighbor_bundles(offset=args.offset)

    model = YOLO("yolo11s.pt")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Running YOLO11s on device={device}, confidence={CONFIDENCE}")

    pools = defaultdict(lambda: {"dets": [], "gts": []})
    timing = defaultdict(lambda: {"preprocess_ms": [], "total_ms": []})
    frame_analysis_rows = []
    debug_counts = defaultdict(int)

    n_prev_missing = n_next_missing = n_prev_rejected = n_next_rejected = 0

    for i, frame in enumerate(dataset):
        filename, stem, video_name = frame["filename"], frame["stem"], frame["video_name"]
        quality_group, image, gts = frame["quality_group"], frame["image"], frame["gts"]
        image_id = f"{video_name}/{stem}"

        if (i + 1) % 10 == 0 or (i + 1) == len(dataset):
            print(f"  [{i + 1}/{len(dataset)}] {filename}")

        preds_by_method = {}

        # --- single-frame filters (reuse restoration.filters) ---
        for method in SINGLE_FRAME_METHODS:
            filter_fn = filters.FILTERS[method]
            start = time.perf_counter()
            restored = filter_fn(image)
            preprocess_ms = (time.perf_counter() - start) * 1000

            preds, yolo_ms = run_yolo(model, device, restored)
            preds_by_method[method] = preds

            for group in ("overall", quality_group):
                key = (method, group)
                pools[key]["dets"].extend((image_id, c, conf, box) for c, conf, box in preds)
                pools[key]["gts"].extend((image_id, c, box) for c, box in gts)
                timing[key]["preprocess_ms"].append(preprocess_ms)
                timing[key]["total_ms"].append(preprocess_ms + yolo_ms)

        # --- temporal methods (shared alignment, two fusion variants) ---
        bundle = bundles.get(filename)
        if bundle is not None:
            align_start = time.perf_counter()
            aligned = restore.align_neighbors(image, bundle)
            align_ms = (time.perf_counter() - align_start) * 1000

            if aligned.prev_report is None:
                n_prev_missing += 1
            elif not aligned.prev_ok:
                n_prev_rejected += 1
                print(f"    [align] {filename}: prev neighbor REJECTED ({aligned.prev_report.reason})")
            if aligned.next_report is None:
                n_next_missing += 1
            elif not aligned.next_ok:
                n_next_rejected += 1
                print(f"    [align] {filename}: next neighbor REJECTED ({aligned.next_report.reason})")

            temporal_results = {}
            for method in TEMPORAL_METHODS:
                fuse_start = time.perf_counter()
                result = restore.fuse(aligned, restore.EVAL_METHOD_TO_FUSE[method])
                fuse_ms = (time.perf_counter() - fuse_start) * 1000
                preprocess_ms = align_ms + fuse_ms

                preds, yolo_ms = run_yolo(model, device, result.restored)
                preds_by_method[method] = preds
                temporal_results[method] = result

                for group in ("overall", quality_group):
                    key = (method, group)
                    pools[key]["dets"].extend((image_id, c, conf, box) for c, conf, box in preds)
                    pools[key]["gts"].extend((image_id, c, box) for c, box in gts)
                    timing[key]["preprocess_ms"].append(preprocess_ms)
                    timing[key]["total_ms"].append(preprocess_ms + yolo_ms)

            # Part 7: raw vs each temporal method, per ground-truth class
            for method in TEMPORAL_METHODS:
                for row in frame_class_comparison(gts, preds_by_method["raw"], preds_by_method[method]):
                    frame_analysis_rows.append({
                        "frame_id": image_id,
                        "video": video_name,
                        "quality_group": quality_group,
                        "method": method,
                        **row,
                    })

            # Part 2: debug panels, a couple per quality group
            if debug_counts[quality_group] < args.debug_per_group:
                debug_counts[quality_group] += 1
                out_path = os.path.join(DEBUG_DIR, f"{video_name}_{stem}_debug.jpg")
                restore.save_debug_panel(bundle, aligned, temporal_results, out_path)
        else:
            print(f"    WARNING: no neighbor bundle for {filename} (source video missing?), skipping temporal methods")

    print(f"\nAlignment summary (offset={args.offset}): "
          f"prev missing={n_prev_missing} rejected={n_prev_rejected} | "
          f"next missing={n_next_missing} rejected={n_next_rejected}")

    rows = compute_metrics_rows(pools, timing)
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {RESULTS_CSV}")

    if frame_analysis_rows:
        with open(FRAME_ANALYSIS_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(frame_analysis_rows[0].keys()))
            writer.writeheader()
            writer.writerows(frame_analysis_rows)
        print(f"Wrote {len(frame_analysis_rows)} rows to {FRAME_ANALYSIS_CSV}")

    print("\n=== SUMMARY (overall, all 84 frames) ===")
    for row in rows:
        if row["quality_group"] == "overall":
            print(f"  {row['method']:28s} P={row['precision']:.3f} R={row['recall']:.3f} "
                  f"mAP50={row['map50']:.3f} mAP50-95={row['map50_95']:.3f} "
                  f"latency={row['total_latency_ms']:.1f}ms fps={row['estimated_fps']:.1f}")


if __name__ == "__main__":
    main()

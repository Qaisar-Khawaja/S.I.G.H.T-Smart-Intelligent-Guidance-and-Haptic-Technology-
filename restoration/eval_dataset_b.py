"""
Dataset B orchestrator: evaluation on your real cane-camera footage.

Unlike eval_dataset_a.py, no synthetic degradation is applied here --
these frames (data/frames_real/video1..7/) are already really degraded
by real motion, real sensor noise, and real lighting, so degrading
them further would just be fake data.

For every real frame x every filter in restoration.filters.FILTERS:
  1. run the filter, timing it (wiener_deconv uses its default kernel,
     since -- unlike Dataset A -- we don't know the true blur amount
     on real footage; that's the realistic, honest way to run it here)
  2. run YOLO, timing it
  3. record detection count + mean confidence (always available)
  4. IF this frame has been promoted to data/real_labels/ (i.e. you've
     hand-corrected it via the annotation/ pipeline), also accumulate
     it into the ground-truth precision/recall/mAP@0.5 pool, both
     overall and broken down by quality_group (clear / moderate_blur /
     severe_blur, from data/frames_real_metadata.csv if present) --
     this gives Dataset B a severity breakdown comparable to Dataset
     A's severity curves.

Four outputs:
  results/results_dataset_b_frames.csv   -- one row per frame x filter
                                             (detections/confidence/latency)
  results/results_dataset_b_summary.csv  -- one row per video x filter
                                             (same, averaged)
  results/results_dataset_b_groundtruth.csv -- one row per filter, using
                                             only promoted/annotated frames
                                             (empty until you promote some)
  results/results_dataset_b_by_quality.csv  -- one row per quality_group x
                                             filter (empty if no metadata
                                             CSV is present)

Usage:
    python -m restoration.eval_dataset_b
"""

import csv
import os
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from restoration import filters, metrics
from restoration.detection_metrics import extract_predictions, load_ground_truth, precision_recall_map

FRAMES_DIR = "data/frames_real"
PROMOTED_LABELS_DIR = "data/real_labels"
METADATA_CSV = "data/frames_real_metadata.csv"
FRAMES_CSV = "results/results_dataset_b_frames.csv"
SUMMARY_CSV = "results/results_dataset_b_summary.csv"
GROUNDTRUTH_CSV = "results/results_dataset_b_groundtruth.csv"
BY_QUALITY_CSV = "results/results_dataset_b_by_quality.csv"


def load_quality_groups():
    """filename -> quality_group, or {} if no metadata CSV exists."""
    if not os.path.exists(METADATA_CSV):
        return {}
    with open(METADATA_CSV) as f:
        return {row["filename"]: row["quality_group"] for row in csv.DictReader(f)}

# Matches main.py's live YOLO_CONFIDENCE. Default 0.25 was too strict
# for this camera's actual mounting angle (see module docstring in
# main.py) -- real detections were landing around 5-10% confidence and
# getting silently dropped. Using the same threshold here as the live
# system means this evaluation reflects what the cane will actually see,
# not an artificially stricter lab setting.
CONFIDENCE = 0.15


def find_ground_truth(video_name, stem, img_w, img_h):
    label_path = os.path.join(PROMOTED_LABELS_DIR, video_name, stem + ".txt")
    if not os.path.exists(label_path):
        return None
    return load_ground_truth(label_path, img_w, img_h)


def main():
    os.makedirs("results", exist_ok=True)
    model = YOLO("yolo11s.pt")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Running YOLO on device={device}")

    video_names = sorted(
        d for d in os.listdir(FRAMES_DIR) if os.path.isdir(os.path.join(FRAMES_DIR, d))
    )

    frame_list = []  # (video_name, fname, stem, image)
    for video_name in video_names:
        video_dir = os.path.join(FRAMES_DIR, video_name)
        for fname in sorted(os.listdir(video_dir)):
            img = cv2.imread(os.path.join(video_dir, fname))
            if img is None:
                continue
            stem = os.path.splitext(fname)[0]
            frame_list.append((video_name, fname, stem, img))

    total = len(frame_list) * len(filters.FILTERS)
    print(f"{len(frame_list)} frames x {len(filters.FILTERS)} filters = {total} runs")

    quality_groups = load_quality_groups()

    frame_rows = []
    gt_pool = {name: {"dets": [], "gts": []} for name in filters.FILTERS}
    gt_pool_by_quality = {}  # (quality_group, filter_name) -> {"dets": [], "gts": []}
    n_annotated = 0

    run_index = 0
    for video_name, fname, stem, img in frame_list:
        h, w = img.shape[:2]
        gts = find_ground_truth(video_name, stem, w, h)
        if gts is not None:
            n_annotated += 1
        quality_group = quality_groups.get(fname)

        for filter_name, filter_fn in filters.FILTERS.items():
            run_index += 1
            if run_index % 100 == 0 or run_index == total:
                print(f"[{run_index}/{total}]")

            restored, preprocess_ms = metrics.timed(filter_fn)(img)

            start = time.perf_counter()
            results = model(restored, verbose=False, device=device, conf=CONFIDENCE)
            yolo_ms = (time.perf_counter() - start) * 1000

            preds = extract_predictions(results[0])
            confidences = [conf for _, conf, _ in preds]

            frame_rows.append({
                "video": video_name,
                "frame": fname,
                "filter": filter_name,
                "n_detections": len(preds),
                "mean_confidence": round(float(np.mean(confidences)), 4) if confidences else 0.0,
                "preprocess_ms": round(preprocess_ms, 3),
                "yolo_ms": round(yolo_ms, 3),
                "total_ms": round(preprocess_ms + yolo_ms, 3),
                "annotated": gts is not None,
            })

            if gts is not None:
                gt_pool[filter_name]["dets"].extend((stem, c, conf, box) for c, conf, box in preds)
                gt_pool[filter_name]["gts"].extend((stem, c, box) for c, box in gts)

                if quality_group is not None:
                    key = (quality_group, filter_name)
                    bucket = gt_pool_by_quality.setdefault(key, {"dets": [], "gts": []})
                    bucket["dets"].extend((stem, c, conf, box) for c, conf, box in preds)
                    bucket["gts"].extend((stem, c, box) for c, box in gts)

    with open(FRAMES_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(frame_rows[0].keys()))
        writer.writeheader()
        writer.writerows(frame_rows)
    print(f"\nWrote {len(frame_rows)} rows to {FRAMES_CSV}")

    summary = {}
    for row in frame_rows:
        key = (row["video"], row["filter"])
        summary.setdefault(key, []).append(row)

    summary_rows = []
    for (video_name, filter_name), rows in sorted(summary.items()):
        summary_rows.append({
            "video": video_name,
            "filter": filter_name,
            "n_frames": len(rows),
            "mean_detections": round(float(np.mean([r["n_detections"] for r in rows])), 3),
            "mean_confidence": round(float(np.mean([r["mean_confidence"] for r in rows])), 4),
            "preprocess_ms": round(float(np.mean([r["preprocess_ms"] for r in rows])), 3),
            "yolo_ms": round(float(np.mean([r["yolo_ms"] for r in rows])), 3),
            "total_ms": round(float(np.mean([r["total_ms"] for r in rows])), 3),
        })

    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {len(summary_rows)} rows to {SUMMARY_CSV}")

    gt_rows = []
    for filter_name, pool in gt_pool.items():
        precision, recall, map50 = precision_recall_map(pool["dets"], pool["gts"])
        gt_rows.append({
            "filter": filter_name,
            "n_annotated_frames": n_annotated,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "map50": round(map50, 4),
        })

    with open(GROUNDTRUTH_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(gt_rows[0].keys()))
        writer.writeheader()
        writer.writerows(gt_rows)

    if n_annotated == 0:
        print(f"Wrote {GROUNDTRUTH_CSV} (empty results -- no frames promoted yet via annotation/promote_labels.py)")
    else:
        print(f"Wrote {GROUNDTRUTH_CSV} using {n_annotated} promoted/annotated frames")

    by_quality_rows = []
    for (quality_group, filter_name), pool in sorted(gt_pool_by_quality.items()):
        precision, recall, map50 = precision_recall_map(pool["dets"], pool["gts"])
        n_frames = len({stem for stem, _, _, _ in pool["dets"]} | {stem for stem, _, _ in pool["gts"]})
        by_quality_rows.append({
            "quality_group": quality_group,
            "filter": filter_name,
            "n_frames": n_frames,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "map50": round(map50, 4),
        })

    if by_quality_rows:
        with open(BY_QUALITY_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(by_quality_rows[0].keys()))
            writer.writeheader()
            writer.writerows(by_quality_rows)
        print(f"Wrote {len(by_quality_rows)} rows to {BY_QUALITY_CSV}")
    else:
        print(f"Skipped {BY_QUALITY_CSV} -- no quality_group metadata found for annotated frames")


if __name__ == "__main__":
    main()

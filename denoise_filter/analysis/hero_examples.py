"""
Mines results/results_dataset_b_frames.csv for concrete before/after
examples: frames where some filter found MORE objects than raw found
on the same frame ("rescues"), and the reverse ("filter made it
worse"). For the top candidates, regenerates actual side-by-side
before/after images with YOLO boxes drawn, ready to drop into the
report.

Usage:
    python -m analysis.hero_examples
"""

import csv
import os
from collections import defaultdict

import cv2
import torch
from ultralytics import YOLO

from restoration import filters

FRAMES_CSV = "results/results_dataset_b_frames.csv"
FRAMES_DIR = "data/frames_real"
OUTPUT_DIR = "results/hero_examples"
TOP_N = 6


def load_frame_filter_counts():
    by_frame = defaultdict(dict)
    with open(FRAMES_CSV) as f:
        for row in csv.DictReader(f):
            key = (row["video"], row["frame"])
            by_frame[key][row["filter"]] = {
                "n_detections": int(row["n_detections"]),
                "mean_confidence": float(row["mean_confidence"]),
            }
    return by_frame


def find_rescues_and_failures(by_frame):
    rescues, failures = [], []
    for (video, frame), per_filter in by_frame.items():
        if "raw" not in per_filter:
            continue
        raw_n = per_filter["raw"]["n_detections"]

        best_filter, best = max(
            ((f, d) for f, d in per_filter.items() if f != "raw"),
            key=lambda item: item[1]["n_detections"],
            default=(None, None),
        )
        if best is None:
            continue

        if best["n_detections"] > raw_n:
            rescues.append((video, frame, best_filter, raw_n, best["n_detections"], best["mean_confidence"]))

        worst_filter, worst = min(
            ((f, d) for f, d in per_filter.items() if f != "raw"),
            key=lambda item: item[1]["n_detections"],
        )
        if raw_n > 0 and worst["n_detections"] == 0:
            failures.append((video, frame, worst_filter, raw_n, worst["n_detections"]))

    rescues.sort(key=lambda r: (r[4] - r[3], r[5]), reverse=True)
    failures.sort(key=lambda r: r[3], reverse=True)
    return rescues, failures


def draw_boxes(img, results, label):
    annotated = img.copy()
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls_name = results.names[int(box.cls[0])]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"{cls_name} {conf:.2f}", (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(annotated, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return annotated


def save_comparison(model, device, video, frame, filter_name, out_dir):
    img = cv2.imread(os.path.join(FRAMES_DIR, video, frame))
    filter_fn = filters.FILTERS[filter_name]
    restored = filter_fn(img)

    raw_results = model(img, verbose=False, device=device, conf=0.15)[0]
    restored_results = model(restored, verbose=False, device=device, conf=0.15)[0]

    raw_annotated = draw_boxes(img, raw_results, "raw")
    restored_annotated = draw_boxes(restored, restored_results, filter_name)

    h = max(raw_annotated.shape[0], restored_annotated.shape[0])
    side_by_side = cv2.hconcat([raw_annotated, restored_annotated])

    out_name = f"{video}_{os.path.splitext(frame)[0]}_{filter_name}.jpg"
    cv2.imwrite(os.path.join(out_dir, out_name), side_by_side)
    return out_name


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    by_frame = load_frame_filter_counts()
    rescues, failures = find_rescues_and_failures(by_frame)

    print(f"Found {len(rescues)} rescue candidates, {len(failures)} failure candidates\n")

    print("=== TOP RESCUES (filter found more than raw) ===")
    for video, frame, filt, raw_n, best_n, conf in rescues[:TOP_N]:
        print(f"{video}/{frame}: raw={raw_n} -> {filt}={best_n} (conf={conf:.2f})")

    print("\n=== FAILURE CASES (raw worked, a filter zeroed it out) ===")
    for video, frame, filt, raw_n, worst_n in failures[:TOP_N]:
        print(f"{video}/{frame}: raw={raw_n} -> {filt}={worst_n}")

    model = YOLO("yolo11s.pt")
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"\nSaving side-by-side images to {OUTPUT_DIR}/ ...")
    for video, frame, filt, raw_n, best_n, conf in rescues[:TOP_N]:
        name = save_comparison(model, device, video, frame, filt, OUTPUT_DIR)
        print(f"  rescue: {name}")

    for video, frame, filt, raw_n, worst_n in failures[:TOP_N]:
        name = save_comparison(model, device, video, frame, filt, OUTPUT_DIR)
        print(f"  failure: {name}")


if __name__ == "__main__":
    main()

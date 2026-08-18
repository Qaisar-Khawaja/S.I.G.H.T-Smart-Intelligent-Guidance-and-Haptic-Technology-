"""
Oracle/union recall analysis across every method already evaluated on
Dataset B: for each ground-truth object, was it correctly detected by
AT LEAST ONE of raw / gaussian / bilateral / wiener_denoise /
wiener_deconv / clahe / temporal_fixed / temporal_quality_weighted?

This answers a different question than any single method's aggregate
recall/mAP: even if raw wins pooled precision/recall (see
eval_dataset_b.py, eval_temporal.py), some ground-truth objects may
only ever get caught by one specific filter or by temporal fusion.
This computes the ceiling recall achievable if you could always pick
the right method per object, and identifies which methods are each
object's *unique* rescuer -- i.e. removing that method loses that
detection entirely, since no other method covers it.

Reuses restoration.eval_temporal.load_dataset for identical frame/GT
loading, restoration.filters.FILTERS for the 6 single-frame methods,
temporal.restore for the 2 temporal methods, and
restoration.detection_metrics.match_gt_to_predictions for per-instance
matching -- so results are directly comparable to both
eval_dataset_b.py and eval_temporal.py.

Outputs:
  results/oracle_union_instances.csv  one row per ground-truth object x
                                       all 8 methods (detected?/confidence),
                                       plus which method(s) caught it
  results/oracle_union_summary.csv    raw recall vs union/oracle recall,
                                       overall + by severity

Usage:
    python -m analysis.oracle_recall
    python -m analysis.oracle_recall \
        --model yolov8n.pt --output-dir results/yolov8n
"""

import argparse
import csv
import os
from collections import defaultdict

import torch
from ultralytics import YOLO

from restoration import filters
from restoration.classes import CLASS_NAMES
from restoration.detection_metrics import extract_predictions, match_gt_to_predictions
from restoration.eval_temporal import CONFIDENCE as DEFAULT_CONFIDENCE, load_dataset
from temporal import neighbors as neighbors_mod
from temporal import restore

INSTANCES_CSV = "results/oracle_union_instances.csv"
SUMMARY_CSV = "results/oracle_union_summary.csv"

SINGLE_FRAME_METHODS = list(filters.FILTERS.keys())  # raw, gaussian, bilateral, wiener_denoise, wiener_deconv, clahe
TEMPORAL_METHODS = ["temporal_fixed", "temporal_quality_weighted"]
ALL_METHODS = SINGLE_FRAME_METHODS + TEMPORAL_METHODS

QUALITY_GROUPS = ["clear", "moderate_blur", "severe_blur"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--offset", type=int, default=1)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--expected-raw-recall", type=float)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    instances_csv = os.path.join(args.output_dir, "oracle_union_instances.csv")
    summary_csv = os.path.join(args.output_dir, "oracle_union_summary.csv")

    print("Loading Dataset B (84 frames)...")
    dataset = load_dataset()

    print(f"Building temporal neighbor bundles (offset={args.offset})...")
    bundles = neighbors_mod.build_neighbor_bundles(offset=args.offset)

    model = YOLO(args.model)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(
        f"Running model={args.model} on device={device}, confidence={args.confidence}, "
        f"{len(ALL_METHODS)} methods x {len(dataset)} frames"
    )

    instance_rows = []

    for i, frame in enumerate(dataset):
        filename, stem, video_name = frame["filename"], frame["stem"], frame["video_name"]
        quality_group, image, gts = frame["quality_group"], frame["image"], frame["gts"]
        image_id = f"{video_name}/{stem}"

        if not gts:
            continue
        if (i + 1) % 20 == 0 or (i + 1) == len(dataset):
            print(f"  [{i + 1}/{len(dataset)}] {filename}")

        per_method_matches = {}  # method -> per-gt [(matched, conf), ...]

        for method in SINGLE_FRAME_METHODS:
            restored = filters.FILTERS[method](image)
            results = model(restored, verbose=False, device=device, conf=args.confidence)
            preds = extract_predictions(results[0])
            per_method_matches[method] = match_gt_to_predictions(gts, preds)

        bundle = bundles.get(filename)
        if bundle is not None:
            aligned = restore.align_neighbors(image, bundle)
            for method in TEMPORAL_METHODS:
                result = restore.fuse(aligned, restore.EVAL_METHOD_TO_FUSE[method])
                results = model(result.restored, verbose=False, device=device, conf=args.confidence)
                preds = extract_predictions(results[0])
                per_method_matches[method] = match_gt_to_predictions(gts, preds)
        else:
            for method in TEMPORAL_METHODS:
                per_method_matches[method] = [(False, None)] * len(gts)

        for gt_idx, (class_id, box) in enumerate(gts):
            row = {
                "model": os.path.basename(args.model),
                "confidence_threshold": args.confidence,
                "neighbor_offset": args.offset,
                "frame_id": image_id,
                "quality_group": quality_group,
                "gt_index": gt_idx,
                "class_id": class_id,
                "class_name": CLASS_NAMES.get(class_id, str(class_id)),
            }
            detected_by = []
            for method in ALL_METHODS:
                matched, conf = per_method_matches[method][gt_idx]
                row[f"{method}_detected"] = matched
                row[f"{method}_confidence"] = round(conf, 4) if conf is not None else ""
                if matched:
                    detected_by.append(method)
            row["any_method_detected"] = len(detected_by) > 0
            row["num_methods_detecting"] = len(detected_by)
            row["detected_by"] = "|".join(detected_by)
            row["unique_rescuer"] = detected_by[0] if len(detected_by) == 1 else ""
            instance_rows.append(row)

    with open(instances_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(instance_rows[0].keys()))
        writer.writeheader()
        writer.writerows(instance_rows)
    print(f"\nWrote {len(instance_rows)} rows to {instances_csv}")

    summary_rows = []
    for group in ["overall"] + QUALITY_GROUPS:
        rows = instance_rows if group == "overall" else [r for r in instance_rows if r["quality_group"] == group]
        if not rows:
            continue
        n = len(rows)
        raw_recall = sum(1 for r in rows if r["raw_detected"]) / n
        union_recall = sum(1 for r in rows if r["any_method_detected"]) / n
        summary_rows.append({
            "model": os.path.basename(args.model),
            "confidence_threshold": args.confidence,
            "neighbor_offset": args.offset,
            "quality_group": group,
            "n_gt_instances": n,
            "raw_recall": round(raw_recall, 4),
            "union_recall": round(union_recall, 4),
            "recall_gain": round(union_recall - raw_recall, 4),
        })

    overall = next(row for row in summary_rows if row["quality_group"] == "overall")
    expected_raw_recall = args.expected_raw_recall
    if expected_raw_recall is None and os.path.basename(args.model) == "yolo11s.pt":
        expected_raw_recall = 0.2616
    if expected_raw_recall is not None and abs(overall["raw_recall"] - expected_raw_recall) > 0.005:
        raise RuntimeError(
            "Raw baseline validation failed: expected approximately "
            f"{expected_raw_recall:.4f}, got {overall['raw_recall']:.4f}"
        )

    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {len(summary_rows)} rows to {summary_csv}")

    print("\n=== Raw recall vs union/oracle recall ===")
    for row in summary_rows:
        print(f"  {row['quality_group']:15s} n={row['n_gt_instances']:>4} "
              f"raw_recall={row['raw_recall']:.3f} union_recall={row['union_recall']:.3f} "
              f"gain={row['recall_gain']:+.3f}")

    unique_counts = defaultdict(int)
    for r in instance_rows:
        if r["unique_rescuer"]:
            unique_counts[r["unique_rescuer"]] += 1

    print("\n=== Unique rescues per method (GT objects ONLY that method caught) ===")
    for method in ALL_METHODS:
        print(f"  {method:28s} {unique_counts.get(method, 0)}")

    n_never = sum(1 for r in instance_rows if not r["any_method_detected"])
    print(f"\nGT objects no method EVER caught: {n_never} / {len(instance_rows)}")


if __name__ == "__main__":
    main()

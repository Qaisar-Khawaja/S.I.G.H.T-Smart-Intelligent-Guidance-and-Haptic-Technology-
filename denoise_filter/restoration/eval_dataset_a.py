"""
Dataset A orchestrator: the controlled synthetic-degradation benchmark.

For every clean image in data/clean/, every degradation condition
(none / noise / blur / low_light / glare, each at mild/medium/severe
severity), and every filter in restoration.filters.FILTERS, this:

  1. degrades the clean image (skipped when degradation == "none",
     which answers "does filtering hurt an already-clean frame?")
  2. runs the filter, timing it
  3. computes PSNR/SSIM against the clean image
  4. runs YOLO on the restored image, timing it
  5. matches YOLO detections (restricted to classes.RELEVANT_CLASSES)
     against the COCO ground-truth boxes in data/clean_labels/ to get
     precision / recall / mAP@0.5

wiener_deconv is a special case: under the "blur" degradation only, it
is called with the exact kernel_size/angle used to generate that
severity's blur, matching its docstring's stated assumption that the
blur kernel is known. Under every other condition it uses its default
kernel (a generic guess), which is expected to do little -- showing
that deconvolution only helps when the degradation is actually blur.

Results are aggregated per (degradation, severity, filter) condition
across all images -- one row per condition, not per image. The model,
confidence threshold, random seed, and output path are explicit so runs
from different YOLO checkpoints cannot silently overwrite one another.

Usage:
    python -m restoration.eval_dataset_a
    python -m restoration.eval_dataset_a \
        --model yolov8n.pt \
        --output-csv results/yolov8n/results_dataset_a.csv \
        --seed 0
"""

import argparse
import csv
import hashlib
import os
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from restoration import degrade, filters, metrics
from restoration.detection_metrics import (
    extract_predictions,
    load_ground_truth,
    precision_recall_map,
)

CLEAN_DIR = "data/clean"
LABEL_DIR = "data/clean_labels"
OUTPUT_CSV = "results/results_dataset_a.csv"
DEFAULT_MODEL = "yolo11s.pt"
DEFAULT_CONFIDENCE = 0.25
DEFAULT_SEED = 0

DEGRADATIONS = {
    "noise": (degrade.gaussian_noise, degrade.NOISE_SEVERITIES),
    "blur": (degrade.motion_blur, degrade.BLUR_SEVERITIES),
    "low_light": (degrade.low_light, degrade.LOW_LIGHT_SEVERITIES),
    "glare": (degrade.bright_glare, degrade.GLARE_SEVERITIES),
}


def apply_filter(filter_name, filter_fn, degraded, deg_name, sev_name):
    """
    Runs one filter, timed. Special-cases wiener_deconv on the "blur"
    condition to use the true kernel for that severity (see module
    docstring); every other combination uses the filter's defaults.
    """
    if filter_name == "wiener_deconv" and deg_name == "blur":
        params = degrade.BLUR_SEVERITIES[sev_name]
        return metrics.timed(filter_fn)(
            degraded, kernel_size=params["kernel_size"], angle=params["angle"]
        )
    return metrics.timed(filter_fn)(degraded)


def build_conditions():
    conditions = [("none", None)]
    for deg_name, (_, severities) in DEGRADATIONS.items():
        for sev_name in severities:
            conditions.append((deg_name, sev_name))
    return conditions


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ultralytics checkpoint path")
    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help="YOLO confidence threshold (default: %(default)s)",
    )
    parser.add_argument(
        "--output-csv",
        default=OUTPUT_CSV,
        help="Result CSV path (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Base seed for synthetic noise (default: %(default)s)",
    )
    return parser.parse_args()


def degradation_seed(base_seed, filename, degradation, severity):
    """Stable per-image seed, shared by every filter in one condition."""
    key = f"{base_seed}|{filename}|{degradation}|{severity}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def main():
    args = parse_args()
    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    model = YOLO(args.model)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(
        f"Running model={args.model} confidence={args.confidence} "
        f"seed={args.seed} device={device}"
    )

    image_files = sorted(f for f in os.listdir(CLEAN_DIR) if f.endswith(".jpg"))

    clean_cache, gt_cache = {}, {}
    for fname in image_files:
        img = cv2.imread(os.path.join(CLEAN_DIR, fname))
        clean_cache[fname] = img
        stem = os.path.splitext(fname)[0]
        gt_cache[fname] = load_ground_truth(
            os.path.join(LABEL_DIR, stem + ".txt"), img.shape[1], img.shape[0]
        )

    conditions = build_conditions()
    total_conditions = len(conditions) * len(filters.FILTERS)
    rows = []
    cond_index = 0

    for deg_name, sev_name in conditions:
        for filter_name, filter_fn in filters.FILTERS.items():
            cond_index += 1
            print(f"[{cond_index}/{total_conditions}] {deg_name}/{sev_name or '-'} + {filter_name}")

            psnr_vals, ssim_vals = [], []
            preprocess_ms_vals, yolo_ms_vals = [], []
            all_dets, all_gts = [], []

            for fname in image_files:
                clean = clean_cache[fname]
                gts = gt_cache[fname]

                if deg_name == "none":
                    degraded = clean
                else:
                    deg_fn, severities = DEGRADATIONS[deg_name]
                    # Re-seeding from condition identity gives every filter the
                    # exact same noisy/low-light input and makes reruns stable.
                    np.random.seed(
                        degradation_seed(args.seed, fname, deg_name, sev_name)
                    )
                    degraded = deg_fn(clean, **severities[sev_name])

                restored, preprocess_ms = apply_filter(
                    filter_name, filter_fn, degraded, deg_name, sev_name
                )

                psnr_vals.append(metrics.psnr(restored, clean))
                ssim_vals.append(metrics.ssim(restored, clean))
                preprocess_ms_vals.append(preprocess_ms)

                start = time.perf_counter()
                results = model(
                    restored,
                    verbose=False,
                    device=device,
                    conf=args.confidence,
                )
                yolo_ms_vals.append((time.perf_counter() - start) * 1000)

                preds = extract_predictions(results[0])
                all_dets.extend((fname, c, conf, box) for c, conf, box in preds)
                all_gts.extend((fname, c, box) for c, box in gts)

            precision, recall, map50 = precision_recall_map(all_dets, all_gts)
            mean_preprocess_ms = float(np.mean(preprocess_ms_vals))
            mean_yolo_ms = float(np.mean(yolo_ms_vals))

            rows.append({
                "model": os.path.basename(args.model),
                "confidence": args.confidence,
                "seed": args.seed,
                "degradation": deg_name,
                "severity": sev_name or "-",
                "filter": filter_name,
                "psnr": round(float(np.mean(psnr_vals)), 3),
                "ssim": round(float(np.mean(ssim_vals)), 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "map50": round(map50, 4),
                "preprocess_ms": round(mean_preprocess_ms, 3),
                "yolo_ms": round(mean_yolo_ms, 3),
                "total_ms": round(mean_preprocess_ms + mean_yolo_ms, 3),
                "n_images": len(image_files),
            })

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()

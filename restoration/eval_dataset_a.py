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
across all images -- one row per condition, not per image -- and
written to results/results_dataset_a.csv.

Usage:
    python -m restoration.eval_dataset_a
"""

import csv
import os
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from restoration import degrade, filters, metrics
from restoration.classes import RELEVANT_CLASSES

CLEAN_DIR = "data/clean"
LABEL_DIR = "data/clean_labels"
OUTPUT_CSV = "results/results_dataset_a.csv"
IOU_THRESHOLD = 0.5

DEGRADATIONS = {
    "noise": (degrade.gaussian_noise, degrade.NOISE_SEVERITIES),
    "blur": (degrade.motion_blur, degrade.BLUR_SEVERITIES),
    "low_light": (degrade.low_light, degrade.LOW_LIGHT_SEVERITIES),
    "glare": (degrade.bright_glare, degrade.GLARE_SEVERITIES),
}


def load_ground_truth(label_path, img_w, img_h):
    """Reads a YOLO-format label file, returns [(class_id, (x1,y1,x2,y2)), ...]."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            class_id = int(parts[0])
            if class_id not in RELEVANT_CLASSES:
                continue
            cx, cy, w, h = (float(p) for p in parts[1:5])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append((class_id, (x1, y1, x2, y2)))
    return boxes


def extract_predictions(yolo_result):
    """Returns [(class_id, confidence, (x1,y1,x2,y2)), ...] for relevant classes only."""
    preds = []
    for box in yolo_result.boxes:
        class_id = int(box.cls[0])
        if class_id not in RELEVANT_CLASSES:
            continue
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        preds.append((class_id, conf, (x1, y1, x2, y2)))
    return preds


def iou(box1, box2):
    xa1, ya1, xa2, ya2 = box1
    xb1, yb1, xb2, yb2 = box2
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def match_detections(detections, ground_truths, iou_threshold=IOU_THRESHOLD):
    """
    detections: [(image_id, class_id, confidence, box), ...]
    ground_truths: [(image_id, class_id, box), ...]

    Greedily matches each detection (highest confidence first) to the
    best remaining unmatched ground-truth box of the same image+class.
    Returns (tp, fp) boolean arrays aligned to detections sorted by
    confidence descending, plus that sorted list.
    """
    detections = sorted(detections, key=lambda d: -d[2])

    gt_by_key = defaultdict(list)
    for image_id, class_id, box in ground_truths:
        gt_by_key[(image_id, class_id)].append({"box": box, "matched": False})

    tp = np.zeros(len(detections))
    fp = np.zeros(len(detections))

    for i, (image_id, class_id, conf, box) in enumerate(detections):
        best_iou, best_gt = 0.0, None
        for gt in gt_by_key.get((image_id, class_id), []):
            if gt["matched"]:
                continue
            current_iou = iou(box, gt["box"])
            if current_iou > best_iou:
                best_iou, best_gt = current_iou, gt
        if best_gt is not None and best_iou >= iou_threshold:
            tp[i] = 1
            best_gt["matched"] = True
        else:
            fp[i] = 1

    return tp, fp, detections


def average_precision(recalls, precisions):
    """VOC-2012-style all-point interpolated AP for a single class."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def precision_recall_map(detections, ground_truths):
    """
    Aggregate precision/recall (at YOLO's own confidence threshold) and
    mAP@0.5 (swept over confidence, averaged across classes present in
    ground_truths) for one (degradation, severity, filter) condition.
    """
    if not ground_truths:
        return 0.0, 0.0, 0.0

    tp, fp, _ = match_detections(detections, ground_truths)
    total_tp, total_fp = tp.sum(), fp.sum()
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / len(ground_truths)

    classes_present = {c for _, c, _ in ground_truths}
    aps = []
    for class_id in classes_present:
        class_gts = [g for g in ground_truths if g[1] == class_id]
        class_dets = [d for d in detections if d[1] == class_id]
        c_tp, c_fp, _ = match_detections(class_dets, class_gts)
        tp_cum = np.cumsum(c_tp)
        fp_cum = np.cumsum(c_fp)
        recalls = tp_cum / len(class_gts)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-10)
        aps.append(average_precision(recalls, precisions))

    map50 = float(np.mean(aps)) if aps else 0.0
    return precision, recall, map50


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


def main():
    os.makedirs("results", exist_ok=True)
    model = YOLO("yolov8n.pt")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Running YOLO on device={device}")

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
                    degraded = deg_fn(clean, **severities[sev_name])

                restored, preprocess_ms = apply_filter(
                    filter_name, filter_fn, degraded, deg_name, sev_name
                )

                psnr_vals.append(metrics.psnr(restored, clean))
                ssim_vals.append(metrics.ssim(restored, clean))
                preprocess_ms_vals.append(preprocess_ms)

                start = time.perf_counter()
                results = model(restored, verbose=False, device=device)
                yolo_ms_vals.append((time.perf_counter() - start) * 1000)

                preds = extract_predictions(results[0])
                all_dets.extend((fname, c, conf, box) for c, conf, box in preds)
                all_gts.extend((fname, c, box) for c, box in gts)

            precision, recall, map50 = precision_recall_map(all_dets, all_gts)
            mean_preprocess_ms = float(np.mean(preprocess_ms_vals))
            mean_yolo_ms = float(np.mean(yolo_ms_vals))

            rows.append({
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

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

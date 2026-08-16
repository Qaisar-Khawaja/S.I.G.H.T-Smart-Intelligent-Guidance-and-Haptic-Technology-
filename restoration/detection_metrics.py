"""
Shared YOLO-detection scoring: ground-truth loading, box matching, and
precision/recall/mAP@0.5. Used by both eval_dataset_a.py (COCO ground
truth) and eval_dataset_b.py (your hand-corrected real-frame ground
truth) so the two datasets are scored identically.
"""

import os
from collections import defaultdict

import numpy as np

from restoration.classes import RELEVANT_CLASSES

IOU_THRESHOLD = 0.5


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


def _map_at_iou(detections, ground_truths, iou_threshold):
    """mAP averaged over classes present in ground_truths, at one IoU threshold."""
    classes_present = {c for _, c, _ in ground_truths}
    aps = []
    for class_id in classes_present:
        class_gts = [g for g in ground_truths if g[1] == class_id]
        class_dets = [d for d in detections if d[1] == class_id]
        c_tp, c_fp, _ = match_detections(class_dets, class_gts, iou_threshold)
        tp_cum = np.cumsum(c_tp)
        fp_cum = np.cumsum(c_fp)
        recalls = tp_cum / len(class_gts)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-10)
        aps.append(average_precision(recalls, precisions))
    return float(np.mean(aps)) if aps else 0.0


# COCO-style IoU sweep for mAP@0.5:0.95.
MAP_5095_THRESHOLDS = np.arange(0.5, 1.0, 0.05)


def precision_recall_map(detections, ground_truths):
    """
    Aggregate precision/recall (at YOLO's own confidence threshold) and
    mAP@0.5 (swept over confidence, averaged across classes present in
    ground_truths) for one condition's pooled detections/ground-truths.
    """
    if not ground_truths:
        return 0.0, 0.0, 0.0

    tp, fp, _ = match_detections(detections, ground_truths)
    total_tp, total_fp = tp.sum(), fp.sum()
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / len(ground_truths)

    map50 = _map_at_iou(detections, ground_truths, IOU_THRESHOLD)
    return precision, recall, map50


def precision_recall_map_full(detections, ground_truths):
    """
    Same as precision_recall_map, plus mAP@0.5:0.95 (mean AP over IoU
    thresholds 0.5, 0.55, ..., 0.95 -- the COCO-style metric), reusing
    the same match_detections/average_precision machinery at each
    threshold. Returns (precision, recall, map50, map50_95).
    """
    if not ground_truths:
        return 0.0, 0.0, 0.0, 0.0

    precision, recall, map50 = precision_recall_map(detections, ground_truths)
    map50_95 = float(np.mean([
        _map_at_iou(detections, ground_truths, t) for t in MAP_5095_THRESHOLDS
    ]))
    return precision, recall, map50, map50_95

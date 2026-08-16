"""
Stage 2 of the annotation pipeline: runs YOLO on every frame in
annotation/to_label/ that doesn't already have a draft label, and
writes YOLO-format .txt boxes (restricted to classes.RELEVANT_CLASSES)
next to each image, plus classes.txt so labelImg shows real class
names.

This is the "AI draft" half of the model-assisted workflow -- these
boxes are NOT ground truth yet. You (the human) open annotation/to_label/
in labelImg next, and fix every box: delete wrong detections, add
missed objects, nudge misaligned ones. Only after that correction pass
do labels count as verified (see promote_labels.py).

Safe to re-run: images that already have a .txt are left untouched, so
it won't overwrite in-progress manual edits.

Usage:
    python -m annotation.prelabel
"""

import os

import cv2
import torch
from ultralytics import YOLO

from restoration.classes import RELEVANT_CLASSES

TO_LABEL_DIR = "annotation/to_label"

# Deliberately lower than main.py's live YOLO_CONFIDENCE (0.15) /
# eval_dataset_b.py's CONFIDENCE. This is a one-time drafting aid, not
# deployed behavior -- its only job is to give you a draggable starting
# box instead of drawing from scratch. Even weak real hints (a chair at
# 6.5% confidence, empirically checked) are worth keeping since fixing
# a box is faster than drawing one; false positives just get deleted.
CONFIDENCE = 0.05


def write_yolo_label(path, detections, img_w, img_h):
    lines = []
    for class_id, box in detections:
        x1, y1, x2, y2 = box
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO("yolo11s.pt")

    classes_path = os.path.join(TO_LABEL_DIR, "classes.txt")
    if not os.path.exists(classes_path):
        with open(classes_path, "w") as f:
            for class_id in range(len(model.names)):
                f.write(model.names[class_id] + "\n")

    image_files = sorted(
        f for f in os.listdir(TO_LABEL_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    drafted = 0
    for fname in image_files:
        stem = os.path.splitext(fname)[0]
        label_path = os.path.join(TO_LABEL_DIR, stem + ".txt")
        if os.path.exists(label_path):
            continue  # don't clobber existing draft/manual edits

        img = cv2.imread(os.path.join(TO_LABEL_DIR, fname))
        h, w = img.shape[:2]

        results = model(img, verbose=False, device=device, conf=CONFIDENCE)
        detections = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            if class_id not in RELEVANT_CLASSES:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append((class_id, (x1, y1, x2, y2)))

        write_yolo_label(label_path, detections, w, h)
        drafted += 1

    print(f"Drafted labels for {drafted} new frames (skipped {len(image_files) - drafted} already-drafted).")
    print(f"Now open {TO_LABEL_DIR}/ in labelImg (format: YOLO) and correct every box.")
    print("When done: python -m annotation.promote_labels")


if __name__ == "__main__":
    main()

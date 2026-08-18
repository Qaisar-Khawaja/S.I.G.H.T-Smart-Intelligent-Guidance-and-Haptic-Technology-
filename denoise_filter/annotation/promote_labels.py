"""
Stage 3 of the annotation pipeline: promotes human-corrected labels
from annotation/to_label/ into data/real_labels/ (organized back into
per-video subfolders), where they become the confirmed ground truth
eval_dataset_b.py reads.

Only run this AFTER you've gone through every image in
annotation/to_label/ in labelImg and fixed the boxes -- this script
has no way to tell a corrected label from an unreviewed AI draft, so
promoting too early would silently make Dataset B's "ground truth"
circular again (see the discussion this pipeline exists to avoid).

Usage:
    python -m annotation.promote_labels
"""

import os
import shutil

TO_LABEL_DIR = "annotation/to_label"
PROMOTED_IMAGES_DIR = "data/real_frames_annotated"
PROMOTED_LABELS_DIR = "data/real_labels"


def main():
    label_files = sorted(f for f in os.listdir(TO_LABEL_DIR) if f.endswith(".txt") and f != "classes.txt")

    promoted = 0
    for label_fname in label_files:
        stem = os.path.splitext(label_fname)[0]
        if "__" not in stem:
            print(f"skipping {label_fname}: doesn't match 'video__frame' naming, can't route it")
            continue
        video_name, _ = stem.split("__", 1)

        image_fname = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = stem + ext
            if os.path.exists(os.path.join(TO_LABEL_DIR, candidate)):
                image_fname = candidate
                break
        if image_fname is None:
            print(f"skipping {label_fname}: no matching image found")
            continue

        image_out_dir = os.path.join(PROMOTED_IMAGES_DIR, video_name)
        label_out_dir = os.path.join(PROMOTED_LABELS_DIR, video_name)
        os.makedirs(image_out_dir, exist_ok=True)
        os.makedirs(label_out_dir, exist_ok=True)

        original_frame_name = stem.split("__", 1)[1] + os.path.splitext(image_fname)[1]
        shutil.copy(os.path.join(TO_LABEL_DIR, image_fname), os.path.join(image_out_dir, original_frame_name))
        shutil.copy(
            os.path.join(TO_LABEL_DIR, label_fname),
            os.path.join(label_out_dir, stem.split("__", 1)[1] + ".txt"),
        )
        promoted += 1

    print(f"Promoted {promoted} confirmed frames to {PROMOTED_IMAGES_DIR}/ and {PROMOTED_LABELS_DIR}/")
    print("These are now ground truth for eval_dataset_b.py.")


if __name__ == "__main__":
    main()

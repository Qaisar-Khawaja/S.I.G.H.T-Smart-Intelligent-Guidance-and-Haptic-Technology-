"""
Stage 1 of the annotation pipeline: picks a stratified batch of real
frames to annotate next and copies them into annotation/to_label/.

Frames already promoted to data/real_labels/ (i.e. already annotated
and confirmed) are skipped automatically, so re-running this script
after finishing a batch pulls in a NEW batch instead of repeating work
-- this is how you "annotate more" later.

Usage:
    python -m annotation.select_subset               # 5 frames/video (35 total)
    python -m annotation.select_subset --per-video 3  # smaller batch
"""

import argparse
import os
import shutil

FRAMES_DIR = "data/frames_real"
PROMOTED_LABELS_DIR = "data/real_labels"
TO_LABEL_DIR = "annotation/to_label"


def already_promoted(video_name, frame_filename):
    stem = os.path.splitext(frame_filename)[0]
    return os.path.exists(os.path.join(PROMOTED_LABELS_DIR, video_name, stem + ".txt"))


def pick_evenly_spaced(items, n):
    if n >= len(items):
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-video", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(TO_LABEL_DIR, exist_ok=True)

    video_names = sorted(
        d for d in os.listdir(FRAMES_DIR)
        if os.path.isdir(os.path.join(FRAMES_DIR, d))
    )

    total_copied = 0
    for video_name in video_names:
        video_dir = os.path.join(FRAMES_DIR, video_name)
        frames = sorted(os.listdir(video_dir))
        available = [f for f in frames if not already_promoted(video_name, f)]

        selected = pick_evenly_spaced(available, args.per_video)

        for fname in selected:
            dest_name = f"{video_name}__{fname}"
            shutil.copy(
                os.path.join(video_dir, fname),
                os.path.join(TO_LABEL_DIR, dest_name),
            )
            total_copied += 1

        print(f"{video_name}: {len(selected)} frames queued "
              f"({len(available)} available, {len(frames) - len(available)} already promoted)")

    print(f"\n{total_copied} frames copied to {TO_LABEL_DIR}/")
    print("Next: python -m annotation.prelabel")


if __name__ == "__main__":
    main()

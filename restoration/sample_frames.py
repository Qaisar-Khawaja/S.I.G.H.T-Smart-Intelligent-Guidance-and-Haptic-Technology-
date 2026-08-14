"""
Extracts a representative subset of frames from each real cane-sweep
video (data/videos/video1.mov .. video7.mov) for Dataset B.

Rather than keeping every frame (a 20s clip at 30fps is 600
near-duplicate frames), this scores every frame by sharpness
(variance of Laplacian, see metrics.sharpness) and samples frames
spread evenly across the observed sharpness range -- so the ~25 frames
kept per video span sharp / moderately-blurred / heavily-blurred,
instead of being biased toward whatever the camera happened to see
most.

Usage:
    python -m restoration.sample_frames
"""

import os

import cv2
import numpy as np

from restoration.metrics import sharpness

VIDEO_DIR = "data/videos"
OUTPUT_DIR = "data/frames_real"
FRAMES_PER_VIDEO = 25
SHARPNESS_BUCKETS = 5  # sharpness range is split into this many bins;
                        # frames are drawn evenly from each bin


def extract_candidate_frames(video_path, stride=3):
    """
    Reads every `stride`-th frame (skip near-duplicates cheaply) and
    returns a list of (frame_index, frame, sharpness_score).
    """
    cap = cv2.VideoCapture(video_path)
    candidates = []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % stride == 0:
            candidates.append((frame_index, frame, sharpness(frame)))
        frame_index += 1

    cap.release()
    return candidates


def select_representative_frames(candidates, n_frames, n_buckets):
    """
    Buckets candidates by sharpness into n_buckets equal-width bins,
    then samples ~n_frames/n_buckets evenly-spaced frames from each
    non-empty bin. Falls back gracefully if a bucket is sparse.
    """
    if not candidates:
        return []

    scores = np.array([c[2] for c in candidates])
    lo, hi = scores.min(), scores.max()

    if hi == lo:
        # Degenerate case: uniform sharpness throughout. Just take an
        # evenly-spaced sample across the whole video.
        indices = np.linspace(0, len(candidates) - 1, n_frames).astype(int)
        return [candidates[i] for i in sorted(set(indices))]

    bucket_edges = np.linspace(lo, hi, n_buckets + 1)
    per_bucket = max(1, n_frames // n_buckets)

    selected = []
    for b in range(n_buckets):
        low, high = bucket_edges[b], bucket_edges[b + 1]
        in_bucket = [c for c in candidates if low <= c[2] <= high]
        if not in_bucket:
            continue
        in_bucket.sort(key=lambda c: c[0])  # keep temporal order
        step = max(1, len(in_bucket) // per_bucket)
        selected.extend(in_bucket[::step][:per_bucket])

    return selected


def process_video(video_path, output_dir, n_frames, n_buckets):
    candidates = extract_candidate_frames(video_path)
    selected = select_representative_frames(candidates, n_frames, n_buckets)

    os.makedirs(output_dir, exist_ok=True)
    for frame_index, frame, score in selected:
        filename = f"frame_{frame_index:05d}_sharp{score:.1f}.jpg"
        cv2.imwrite(os.path.join(output_dir, filename), frame)

    return len(selected)


def main():
    video_files = sorted(f for f in os.listdir(VIDEO_DIR) if f.endswith(".mov"))

    for video_file in video_files:
        video_name = os.path.splitext(video_file)[0]
        video_path = os.path.join(VIDEO_DIR, video_file)
        output_dir = os.path.join(OUTPUT_DIR, video_name)

        count = process_video(
            video_path, output_dir, FRAMES_PER_VIDEO, SHARPNESS_BUCKETS
        )
        print(f"{video_file}: saved {count} frames -> {output_dir}")


if __name__ == "__main__":
    main()

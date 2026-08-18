"""
Part 1: reconstructs I(t-offset), I(t), I(t+offset) neighbor triples for
every annotated Dataset B frame, pulling neighbors from the original
source video (data/videos/videoN.mov) rather than from data/frames_real/
(which only holds the 84 already-selected target frames).

I(t) itself is never re-decoded from video -- restoration/eval_temporal.py
loads it straight from data/frames_real/, the exact pixels the ground
truth in data/real_labels/ was drawn against. Only the neighbors come
from a fresh video decode.
"""

import csv
import os
import re
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

METADATA_CSV = "data/frames_real_metadata.csv"
VIDEO_DIR = "data/videos"

DEFAULT_OFFSET = 1


def load_metadata(metadata_csv=METADATA_CSV):
    """filename -> metadata dict, numeric fields parsed."""
    with open(metadata_csv) as f:
        rows = {}
        for row in csv.DictReader(f):
            row["frame_number"] = int(row["frame_number"])
            row["timestamp_sec"] = float(row["timestamp_sec"])
            row["fps"] = float(row["fps"])
            row["width"] = int(row["width"])
            row["height"] = int(row["height"])
            rows[row["filename"]] = row
        return rows


def video_path_for(video_id, video_dir=VIDEO_DIR):
    """
    'v01_20-16' -> 'data/videos/video1.mov'. The numeric prefix in
    video_id lines up positionally with data/videos/videoN.mov (checked
    against each video's actual ffprobe duration -- there's no separate
    mapping file, so this is the one place that link is encoded).
    """
    match = re.match(r"v(\d+)_", video_id)
    n = int(match.group(1))
    return os.path.join(video_dir, f"video{n}.mov")


@dataclass
class NeighborBundle:
    filename: str
    video_name: str
    base_idx: int
    total_frames: int
    offset: int
    prev_frame: Optional[np.ndarray]
    next_frame: Optional[np.ndarray]


def _scan_timestamps(video_path):
    """One sequential decode pass; returns POS_MSEC per frame index."""
    cap = cv2.VideoCapture(video_path)
    timestamps = []
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
    cap.release()
    return timestamps


def _nearest_index(timestamps, target_sec):
    target_ms = target_sec * 1000.0
    return min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - target_ms))


def _collect_frames(video_path, needed_indices):
    """Second sequential pass; returns {idx: BGR frame} for idx in needed_indices."""
    needed = {i for i in needed_indices if i is not None and i >= 0}
    if not needed:
        return {}
    cap = cv2.VideoCapture(video_path)
    found = {}
    idx = 0
    max_needed = max(needed)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in needed:
            found[idx] = frame
        if idx >= max_needed:
            break
        idx += 1
    cap.release()
    return found


def build_neighbor_bundles(offset=DEFAULT_OFFSET, metadata_csv=METADATA_CSV, video_dir=VIDEO_DIR):
    """
    Locates every annotated frame's true position in its source video
    and gathers the raw prev/next neighbor frames `offset` frames away.

    Why match on timestamp_sec instead of trusting frame_number
    directly: frame_number in the metadata CSV was computed as
    round(timestamp_sec * fps), which assumes constant frame rate.
    video1.mov is variable-frame-rate in practice, so frame_number can
    land up to ~6 frames away from where OpenCV's sequential decode
    actually is at that timestamp (verified empirically; video2/video3
    matched exactly since they're CFR). Matching each candidate frame's
    own presentation timestamp (CAP_PROP_POS_MSEC) against the target
    timestamp is robust to that, at the cost of one extra sequential
    decode pass per video -- cheap at this frame count.

    Returns {filename: NeighborBundle}. Missing neighbors (video start/
    end, or a video file that's absent) are represented as None fields,
    not raised errors -- callers (temporal/restore.py) handle that as a
    normal boundary case.
    """
    metadata = load_metadata(metadata_csv)

    by_video = {}
    for filename, row in metadata.items():
        by_video.setdefault(row["video_id"], []).append((filename, row))

    bundles = {}
    for video_id, rows in sorted(by_video.items()):
        video_path = video_path_for(video_id, video_dir)
        if not os.path.exists(video_path):
            print(f"  [neighbors] WARNING: {video_path} not found, skipping {len(rows)} frames")
            continue

        timestamps = _scan_timestamps(video_path)
        total_frames = len(timestamps)
        video_num = int(re.match(r"v(\d+)_", video_id).group(1))
        video_name = f"video{video_num}"

        base_indices = {
            filename: _nearest_index(timestamps, row["timestamp_sec"])
            for filename, row in rows
        }

        needed = set()
        for base_idx in base_indices.values():
            needed.add(base_idx - offset)
            needed.add(base_idx + offset)
        frames = _collect_frames(video_path, needed)

        for filename, row in rows:
            base_idx = base_indices[filename]
            prev_idx, next_idx = base_idx - offset, base_idx + offset
            bundles[filename] = NeighborBundle(
                filename=filename,
                video_name=video_name,
                base_idx=base_idx,
                total_frames=total_frames,
                offset=offset,
                prev_frame=frames.get(prev_idx) if prev_idx >= 0 else None,
                next_frame=frames.get(next_idx) if next_idx < total_frames else None,
            )

    return bundles

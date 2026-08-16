"""
Part 8: mines results/temporal_frame_analysis.csv for concrete
before/after examples of temporal restoration vs raw, and saves
side-by-side (RAW | TEMPORAL) images with YOLO boxes drawn, grouped by
outcome:

  results/temporal_examples/rescued/   ground-truth objects raw missed
                                        that a temporal method caught
  results/temporal_examples/degraded/  objects raw caught that a
                                        temporal method lost (includes
                                        at least one failure case, not
                                        cherry-picked away)
  results/temporal_examples/neutral/   objects both methods agree on
                                        (no_change outcome) -- included
                                        so the "rescued"/"degraded"
                                        folders aren't presented as if
                                        they were the typical case

Reuses restoration.filters / temporal.restore for reconstructing the
exact restored image, and analysis.hero_examples.draw_boxes for
consistent box-drawing style with the existing Dataset B hero examples.

Usage:
    python -m analysis.temporal_hero_examples
"""

import csv
import os

import cv2
import torch
from ultralytics import YOLO

from analysis.hero_examples import draw_boxes
from restoration.classes import CLASS_NAMES
from restoration.eval_temporal import CONFIDENCE, FRAMES_DIR
from temporal import neighbors as neighbors_mod
from temporal import restore

FRAME_ANALYSIS_CSV = "results/temporal_frame_analysis.csv"
OUTPUT_ROOT = "results/temporal_examples"
N_PER_OUTCOME = 6

# Priority classes called out in the project brief -- preferred when
# there are more candidates than slots.
PRIORITY_CLASS_NAMES = {"person", "chair", "backpack", "bottle", "dining table"}


def load_rows():
    with open(FRAME_ANALYSIS_CSV) as f:
        return list(csv.DictReader(f))


def select_examples(rows, outcome, n, severity_priority=("severe_blur", "moderate_blur", "clear")):
    candidates = [r for r in rows if r["outcome"] == outcome]

    def sort_key(r):
        class_name = CLASS_NAMES.get(int(r["class_id"]), "")
        is_priority_class = class_name not in PRIORITY_CLASS_NAMES  # False sorts first
        sev_rank = severity_priority.index(r["quality_group"]) if r["quality_group"] in severity_priority else 99
        return (is_priority_class, sev_rank)

    candidates.sort(key=sort_key)

    seen_frames, selected = set(), []
    for r in candidates:
        if r["frame_id"] in seen_frames:
            continue
        seen_frames.add(r["frame_id"])
        selected.append(r)
        if len(selected) >= n:
            break
    return selected


def build_comparison(model, device, row, bundles):
    frame_id, method = row["frame_id"], row["method"]
    video_name, stem = frame_id.split("/", 1)
    # frame_id was built as f"{video_name}/{stem}" (no extension) in eval_temporal.py;
    # bundles is keyed by the actual filename (with extension).
    matches = [f for f in bundles if os.path.splitext(f)[0] == stem]
    if not matches:
        return None
    actual_filename = matches[0]

    target = cv2.imread(os.path.join(FRAMES_DIR, video_name, actual_filename))
    if target is None:
        return None

    bundle = bundles[actual_filename]
    aligned = restore.align_neighbors(target, bundle)
    result = restore.fuse(aligned, restore.EVAL_METHOD_TO_FUSE[method])

    raw_results = model(target, verbose=False, device=device, conf=CONFIDENCE)[0]
    restored_results = model(result.restored, verbose=False, device=device, conf=CONFIDENCE)[0]

    class_name = CLASS_NAMES.get(int(row["class_id"]), str(row["class_id"]))
    info = (f"{row['outcome']} | {class_name} | {row['quality_group']} | "
            f"{frame_id} | {method}")

    raw_annotated = draw_boxes(target, raw_results, "RAW")
    restored_annotated = draw_boxes(result.restored, restored_results, method)

    banner_h = 24
    banner = cv2.copyMakeBorder(
        cv2.hconcat([raw_annotated, restored_annotated]),
        banner_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30),
    )
    cv2.putText(banner, info, (6, banner_h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return banner


def main():
    rows = load_rows()
    if not rows:
        print(f"{FRAME_ANALYSIS_CSV} is empty -- run restoration/eval_temporal.py first")
        return

    print("Building temporal neighbor bundles...")
    bundles = neighbors_mod.build_neighbor_bundles(offset=1)

    model = YOLO("yolo11s.pt")
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    outcome_to_folder = {"rescued": "rescued", "lost": "degraded", "no_change": "neutral"}

    for outcome, folder in outcome_to_folder.items():
        out_dir = os.path.join(OUTPUT_ROOT, folder)
        os.makedirs(out_dir, exist_ok=True)

        examples = select_examples(rows, outcome, N_PER_OUTCOME)
        print(f"\n{outcome} -> {folder}/: {len(examples)} example(s) selected "
              f"(of {sum(1 for r in rows if r['outcome'] == outcome)} candidates)")

        for row in examples:
            panel = build_comparison(model, device, row, bundles)
            if panel is None:
                print(f"  skipped {row['frame_id']} ({row['method']}): couldn't rebuild frame")
                continue
            video_name, stem = row["frame_id"].split("/", 1)
            out_name = f"{video_name}_{stem}_{row['method']}_{CLASS_NAMES.get(int(row['class_id']), row['class_id'])}.jpg"
            cv2.imwrite(os.path.join(out_dir, out_name), panel)
            print(f"  saved {out_name}")

    print(f"\nDone. Examples written under {OUTPUT_ROOT}/{{rescued,degraded,neutral}}/")


if __name__ == "__main__":
    main()

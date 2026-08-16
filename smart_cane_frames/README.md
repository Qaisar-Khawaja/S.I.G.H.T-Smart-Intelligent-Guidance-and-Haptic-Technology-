# Smart Cane Real-Camera Evaluation Frames

This folder contains 84 lossless PNG frames selected from the 7 provided cane-sweep videos (12 frames per video).

## Contents
- `frames/`: flat folder of evaluation images, easy to use with OpenCV/YOLO/annotation tools.
- `metadata.csv`: source video, frame number, timestamp, coarse quality group, lighting note, Laplacian sharpness score, and brightness.
- `contact_sheets/`: overview sheets used to review temporal coverage.
- `extract_selected_frames.py`: reproducible extraction script (paths can be changed for another machine).

## Selection strategy
Frames were intentionally sampled across the video timeline and coarsely grouped as:
- `clear`: relatively informative / less degraded moments
- `moderate_blur`: visible motion degradation but potentially recoverable
- `severe_blur`: challenging motion-blurred frames

The quality groups are coarse evaluation strata, not ground-truth physical blur measurements. `laplacian_variance` is included as an objective sharpness proxy, but it is scene-dependent and should not be treated as a universal blur threshold.

Video `v05_20-23` also contains darker/occluded frames; those are marked `dim_or_occluded` in the metadata.

## Recommended use
1. Annotate only the COCO classes you intend to evaluate (e.g. person, chair, backpack, bottle, dining table).
2. Run the exact same annotated frames through every preprocessing method.
3. Report overall metrics AND metrics by `quality_group`.
4. Keep `raw` as the no-preprocessing control.
5. Do not claim all 84 frames are statistically independent; nearby video frames originate from the same 7 recordings.

PNG is used to avoid introducing extra lossy compression during the restoration experiment.

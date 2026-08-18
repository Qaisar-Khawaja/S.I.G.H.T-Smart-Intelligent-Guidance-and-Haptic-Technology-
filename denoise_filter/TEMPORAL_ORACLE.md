# Temporal Detection Oracle (Phase 1)

## Motivation

Pixel-level restoration did not improve Dataset B over raw YOLO11s, so this experiment tests whether correct object detections exist in nearby raw video frames and could theoretically bridge target-frame misses at the detection/state level.

## Dataset and baseline

The evaluation uses all 84 annotated target frames (344 scored GT instances) from the seven original cane-camera videos. Target-frame detections use the exact annotated PNGs. YOLO11s (`yolo11s.pt`), confidence 0.15, the repository's relevant-class filter, and same-class IoU >= 0.5 matching are unchanged from Dataset B.
The ±1, ±3, and ±5 windows correspond to approximately ±33 ms, ±100 ms, and ±167 ms at 30 FPS (video 1 metadata reports 29.7 FPS; the others report 30 FPS).

## Association method

Each target-frame GT box seeds independent forward and backward tracklets through the cached full-video detections. Association is same-class and one-to-one, with gates on normalized center displacement, area ratio, and aspect-ratio change. A constant-velocity estimate is used only after a first association; otherwise the GT/last box is held during short gaps. Near-tied assignments (score margin <= 0.06) are flagged as ambiguous, do not update a tracklet, and are not credited as recoveries. This is intentionally conservative where several people or chairs are present.

## Results

| Group | GT | Raw recall | Oracle ±1 | Oracle ±3 | Oracle ±5 | Raw misses | Recoverable ±1 | Recoverable ±3 | Recoverable ±5 | Never recovered ±5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 344 | 0.2616 | 0.3459 | 0.4390 | 0.4826 | 254 | 29 | 61 | 76 | 178 |
| Clear | 125 | 0.3760 | 0.4640 | 0.5280 | 0.5600 | 78 | 11 | 19 | 23 | 55 |
| Moderate blur | 154 | 0.2403 | 0.3377 | 0.4610 | 0.5260 | 117 | 15 | 34 | 44 | 73 |
| Severe blur | 65 | 0.0923 | 0.1385 | 0.2154 | 0.2308 | 59 | 3 | 8 | 9 | 50 |

## Severe-blur focus

Severe blur contains 65 GT instances: 6 raw detections and 59 raw misses. Of those misses, 3 are recoverable within ±1, 8 within ±3, and 9 within ±5. 50 remain unrecovered within ±5; 9 of those have only ambiguous nearby candidates.

## Phase 2 decision

The conservative oracle shows meaningful short-window headroom, so Phase 2 tracking/persistence is justified.

This oracle is an upper-bound diagnostic, not a deployable tracker: it uses the target-frame GT box to seed identity and deliberately excludes ambiguous cases. A real tracker can recover only a subset and must also be judged on precision and stale-track false positives.

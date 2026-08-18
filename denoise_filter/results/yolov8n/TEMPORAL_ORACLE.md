# Temporal Detection Oracle (Phase 1)

## Motivation

Pixel-level restoration did not reliably improve Dataset B over raw yolov8n, so this experiment tests whether correct object detections exist in nearby raw video frames and could theoretically bridge target-frame misses at the detection/state level.

## Dataset and baseline

The evaluation uses all 84 annotated target frames (344 scored GT instances) from the seven original cane-camera videos. Target-frame detections use the exact annotated PNGs. `yolov8n.pt`, confidence 0.15, the repository's relevant-class filter, and same-class IoU >= 0.5 matching are unchanged from Dataset B.
The ±1, ±3, and ±5 windows correspond to approximately ±33 ms, ±100 ms, and ±167 ms at 30 FPS (video 1 metadata reports 29.7 FPS; the others report 30 FPS).

## Association method

Each target-frame GT box seeds independent forward and backward tracklets through the cached full-video detections. Association is same-class and one-to-one, with gates on normalized center displacement, area ratio, and aspect-ratio change. A constant-velocity estimate is used only after a first association; otherwise the GT/last box is held during short gaps. Near-tied assignments (score margin <= 0.06) are flagged as ambiguous, do not update a tracklet, and are not credited as recoveries. This is intentionally conservative where several people or chairs are present.

## Results

| Group | GT | Raw recall | Oracle ±1 | Oracle ±3 | Oracle ±5 | Raw misses | Recoverable ±1 | Recoverable ±3 | Recoverable ±5 | Never recovered ±5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 344 | 0.0669 | 0.1076 | 0.1424 | 0.1715 | 321 | 14 | 26 | 36 | 285 |
| Clear | 125 | 0.1120 | 0.1840 | 0.2480 | 0.2880 | 111 | 9 | 17 | 22 | 89 |
| Moderate blur | 154 | 0.0519 | 0.0844 | 0.1039 | 0.1364 | 146 | 5 | 8 | 13 | 133 |
| Severe blur | 65 | 0.0154 | 0.0154 | 0.0308 | 0.0308 | 64 | 0 | 1 | 1 | 63 |

## Severe-blur focus

Severe blur contains 65 GT instances: 1 raw detections and 64 raw misses. Of those misses, 0 are recoverable within ±1, 1 within ±3, and 1 within ±5. 63 remain unrecovered within ±5; 1 of those have only ambiguous nearby candidates.

## Phase 2 decision

The conservative oracle shows meaningful short-window headroom, so Phase 2 tracking/persistence is justified.

This oracle is an upper-bound diagnostic, not a deployable tracker: it uses the target-frame GT box to seed identity and deliberately excludes ambiguous cases. A real tracker can recover only a subset and must also be judged on precision and stale-track false positives.

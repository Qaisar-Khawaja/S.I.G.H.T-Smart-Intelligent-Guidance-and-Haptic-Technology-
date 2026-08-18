# YOLOv8n Validation

## Purpose

The original smart-cane prototype loaded `yolov8n.pt`, while the retained
Dataset A/B, temporal, oracle, and tracking reports use YOLO11s. This validation
reruns the two image-restoration benchmarks with the original YOLOv8n detector
without overwriting the YOLO11s outputs.

This is a detector validation, not a YOLO version versus restoration factorial
experiment. Within each benchmark, the checkpoint and confidence threshold are
held fixed across raw and filtered inputs.

## Configuration

| Setting | Dataset A | Dataset B |
| --- | --- | --- |
| Model | `yolov8n.pt` | `yolov8n.pt` |
| Model SHA-256 | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` | same |
| Confidence | 0.25 | 0.15 |
| Images/frames | 40 clean source images | 84 annotated target frames |
| Synthetic seed | 0 | not applicable |
| Device in this run | CPU | CPU |

Dataset A uses a stable seed derived from the base seed, source filename,
degradation, and severity. Consequently, every filter sees exactly the same
synthetically degraded input for a given condition.

## Dataset A

Selected mAP@0.5 results:

| Condition | Raw | Best tested method | Best mAP@0.5 |
| --- | ---: | --- | ---: |
| Clean | 0.4255 | Raw | 0.4255 |
| Strong Gaussian noise | 0.0860 | Gaussian | 0.1675 |
| Mild motion blur | 0.1542 | Matched Wiener deconvolution | 0.3178 |
| Medium motion blur | 0.0677 | Matched Wiener deconvolution | 0.1998 |
| Severe motion blur | 0.0464 | Matched Wiener deconvolution | 0.0906 |
| Medium low light | 0.1375 | Bilateral | 0.2166 |
| Severe low light | 0.0005 | Bilateral | 0.0027 |

The synthetic benchmark preserves the earlier qualitative conclusion: a filter
can help when its assumptions match a controlled degradation, but applying a
filter to a clean image does not improve the detector.

## Dataset B

Overall results on real cane-camera frames:

| Method | Precision | Recall | mAP@0.5 |
| --- | ---: | ---: | ---: |
| Raw | 0.7667 | 0.0669 | 0.0504 |
| Gaussian | 0.7419 | 0.0669 | 0.0338 |
| Bilateral | 0.6400 | 0.0465 | 0.0279 |
| Wiener denoising | 0.7037 | 0.0552 | 0.0300 |
| Wiener deconvolution | 0.0000 | 0.0000 | 0.0000 |
| CLAHE | 0.5686 | 0.0843 | 0.0510 |

CLAHE raises recall by 0.0174 and mAP@0.5 by only 0.0006 while reducing
precision by 0.1981. This is not a compelling universal-preprocessing gain.
Raw YOLOv8n remains preferable when precision and simplicity matter.

Raw recall by quality group is 0.1120 for clear frames, 0.0519 for moderate
blur, and 0.0154 for severe blur. The corresponding raw mAP@0.5 values are
0.0838, 0.0705, and 0.0118.

For context, raw YOLO11s on the identical Dataset B frames produced precision
0.8738, recall 0.2616, and mAP@0.5 0.2591. This paired result shows that model
choice materially affects absolute performance. It does not invalidate the
within-model filter comparison.

## Runtime

This run used the current CPU environment. Mean raw YOLO inference time was
62.779 ms on Dataset B. Dataset A raw-condition inference averaged 88.866 ms;
the difference reflects the distinct image sets and runtime state, so these
figures should not be treated as a hardware deployment benchmark.

## Temporal oracle

YOLOv8n detections were also cached for every decoded frame of all seven source
videos. The same conservative GT-seeded association oracle was then evaluated:

| Window | Overall recall | Additional recoveries over raw |
| --- | ---: | ---: |
| Raw target frame | 0.0669 | 0 |
| Past 1 | 0.0814 | 5 |
| Past 3 | 0.1017 | 12 |
| Past 5 | 0.1134 | 16 |
| Bidirectional ±5 | 0.1715 | 36 |

The causal oracle shows that past frames contain some usable detections, but
the severe-blur ceiling is very low: severe-blur recall remains 0.0154 at past
1 and reaches only 0.0308 at past 3/5.

## Tracking and bounded persistence

ByteTrack and BoT-SORT were run continuously on the YOLOv8n full-video
detections. Standard tracker output was compared with a safety-preserving
wrapper that passes every current raw detection through and adds unmatched,
decaying Kalman predictions for at most 1, 3, or 5 frames.

| Method | Precision | Recall | mAP@0.5 | Rescued | Lost | Stale persisted FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw YOLOv8n | 0.7667 | 0.0669 | 0.0504 | 0 | 0 | 0 |
| Standard ByteTrack | 1.0000 | 0.0378 | 0.0244 | 0 | 10 | 0 |
| Standard BoT-SORT | 1.0000 | 0.0407 | 0.0251 | 0 | 9 | 0 |
| ByteTrack persistence 1 | 0.7742 | 0.0698 | 0.0511 | 1 | 0 | 0 |
| BoT-SORT persistence 3 | 0.7812 | 0.0727 | 0.0518 | 2 | 0 | 0 |

The configured selection rule chooses BoT-SORT persistence 3. It recovers two
of the twelve recoveries available in the past-3 oracle, or 16.7% of causal
headroom. Both rescues occur on clear frames: moderate- and severe-blur recall
do not improve. This is therefore a modest continuity result, not evidence that
tracking solves the primary blur failure.

On the current CPU benchmark, raw YOLOv8n runs at 18.98 FPS, ByteTrack
persistence at 18.75 FPS, and BoT-SORT persistence at 16.89 FPS. ByteTrack
persistence 1 is a lower-overhead alternative if deployment cost is weighted
more heavily than the second clear-frame rescue.

## Pixel-level temporal fusion

The pixel experiment aligns the immediately previous and next video frames to
the annotated target with dense Farneback flow, then applies fixed or
sharpness-weighted fusion. Both variants underperform raw YOLOv8n overall:

| Method | Precision | Recall | mAP@0.5 | Latency (ms) |
| --- | ---: | ---: | ---: | ---: |
| Raw | 0.7667 | 0.0669 | 0.0504 | 89.667 |
| Temporal fixed | 0.7000 | 0.0610 | 0.0402 | 179.929 |
| Temporal quality-weighted | 0.6786 | 0.0552 | 0.0377 | 182.334 |

Fixed fusion rescues two frame-class groups and loses four. Quality-weighted
fusion rescues two and loses five, including the only severe-blur raw
detection. Pixel fusion therefore adds substantial latency without a net
detection benefit.

## Filter-union oracle

The eight-method union combines raw, five filters, and both temporal-fusion
variants as an unavailable perfect per-object selector:

| Group | Raw recall | Union recall | Gain |
| --- | ---: | ---: | ---: |
| Overall | 0.0669 | 0.1105 | +0.0436 |
| Clear | 0.1120 | 0.2080 | +0.0960 |
| Moderate blur | 0.0519 | 0.0714 | +0.0195 |
| Severe blur | 0.0154 | 0.0154 | +0.0000 |

The apparent upper-bound gain is concentrated in clear frames. Fully 306 of
344 GT objects are missed by every method, and the union provides no additional
severe-blur detection. This does not justify deploying all filters; it shows
only that their errors are partly complementary.

## Quantitative coverage

All core quantitative experiment families now have retained YOLOv8n outputs:
Dataset A, Dataset B, pixel-level temporal fusion, the filter-union oracle, the
temporal detection oracle, and detection-level tracking/persistence. The
YOLO11s artifacts remain useful as a separate detector comparison and should
not be pooled numerically with YOLOv8n results.

## Reproduction

```bash
MPLCONFIGDIR=/private/tmp/sight-mpl-cache \
  python -m restoration.eval_dataset_a \
  --model yolov8n.pt \
  --confidence 0.25 \
  --seed 0 \
  --output-csv results/yolov8n/results_dataset_a.csv

MPLCONFIGDIR=/private/tmp/sight-mpl-cache \
  python -m restoration.eval_dataset_b \
  --model yolov8n.pt \
  --confidence 0.15 \
  --output-dir results/yolov8n

MPLCONFIGDIR=/private/tmp/sight-mpl-cache \
  python -m restoration.eval_temporal \
  --model yolov8n.pt \
  --confidence 0.15 \
  --offset 1 \
  --output-dir results/yolov8n

MPLCONFIGDIR=/private/tmp/sight-mpl-cache \
  python -m analysis.oracle_recall \
  --model yolov8n.pt \
  --confidence 0.15 \
  --offset 1 \
  --output-dir results/yolov8n \
  --expected-raw-recall 0.0669

python -m analysis.temporal_oracle \
  --model yolov8n.pt \
  --confidence 0.15 \
  --output-dir results/yolov8n \
  --report results/yolov8n/TEMPORAL_ORACLE.md

python -m analysis.temporal_oracle_causal \
  --instances-csv results/yolov8n/temporal_oracle_instances.csv \
  --output-csv results/yolov8n/temporal_oracle_causal_summary.csv

MPLCONFIGDIR=/private/tmp/sight-mpl-cache \
  python -m tracking.evaluate \
  --model yolov8n.pt \
  --confidence 0.15 \
  --cache-dir results/yolov8n/temporal_oracle_cache \
  --causal-oracle results/yolov8n/temporal_oracle_causal_summary.csv \
  --output-dir results/yolov8n \
  --report results/yolov8n/TRACKING_PERSISTENCE.md
```

Outputs are retained under `results/yolov8n/`.

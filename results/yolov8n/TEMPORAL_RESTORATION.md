# YOLOv8n Pixel Fusion and Filter-Union Oracle

## Method

The pixel-level temporal experiment uses the exact 84 annotated target PNGs
from Dataset B and decodes the immediately previous and next frames from the
source videos. Dense Farneback optical flow warps each neighbor toward the
target. Two fusion rules are evaluated:

- fixed weights of 0.25/0.50/0.25 for previous/target/next;
- quality weights derived from the square root of Laplacian variance, with a
  minimum target weight of 0.30.

The existing validity gate requires at least 60% valid warped pixels, mean
absolute difference no greater than 40, SSIM of at least 0.25, and mean flow
magnitude no greater than 60 pixels. No available neighbor was rejected in this
run; six previous neighbors were unavailable at video boundaries.

All methods use `yolov8n.pt`, confidence 0.15, the same relevant-class filter,
and same-class IoU >= 0.5 scoring. The target is always the exact annotated PNG;
only its neighbors come from continuous video decoding.

## Pixel-fusion results

| Method | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Latency (ms) | FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw | 0.7667 | 0.0669 | 0.0504 | 0.0467 | 89.667 | 11.15 |
| Wiener denoising | 0.7037 | 0.0552 | 0.0300 | 0.0279 | 123.293 | 8.11 |
| Wiener deconvolution | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 149.435 | 6.69 |
| CLAHE | 0.5686 | 0.0843 | 0.0510 | 0.0443 | 78.625 | 12.72 |
| Temporal fixed | 0.7000 | 0.0610 | 0.0402 | 0.0361 | 179.929 | 5.56 |
| Temporal quality-weighted | 0.6786 | 0.0552 | 0.0377 | 0.0332 | 182.334 | 5.48 |

Fixed fusion rescues two frame-class groups but loses four; quality-weighted
fusion rescues two and loses five. Fixed fusion does not improve any
moderate- or severe-blur frame-class group. Quality weighting has one
moderate-blur rescue but also loses the only severe-blur raw detection.

For severe blur, raw and fixed-fusion recall are both 0.0154 and mAP@0.5 is
0.0118. Quality-weighted fusion has zero precision, recall, and mAP. Pixel
fusion therefore does not solve the real-blur failure and approximately doubles
latency in this run.

## Eight-method filter union

The union oracle asks whether each GT instance is detected by at least one of
raw, Gaussian, bilateral, Wiener denoising, Wiener deconvolution, CLAHE,
temporal fixed, or temporal quality-weighted input. It is an upper bound that
assumes an unavailable perfect selector; it is not deployable performance.

| Group | GT instances | Raw recall | Union recall | Gain |
| --- | ---: | ---: | ---: | ---: |
| Overall | 344 | 0.0669 | 0.1105 | +0.0436 |
| Clear | 125 | 0.1120 | 0.2080 | +0.0960 |
| Moderate blur | 154 | 0.0519 | 0.0714 | +0.0195 |
| Severe blur | 65 | 0.0154 | 0.0154 | +0.0000 |

Of 344 GT objects, 306 are missed by every method. Unique detections include
seven from CLAHE, two from Gaussian filtering, and two from quality-weighted
temporal fusion. Temporal fixed has no unique detections. Some additional union
recoveries are shared by multiple non-raw methods and therefore are not counted
as unique rescues.

## Interpretation

The YOLOv8n result matches the central conclusion of the earlier YOLO11s
experiment. Temporal pixel fusion can change which individual objects are
detected, but destructive alignment/fusion effects outweigh its rescues in
aggregate. The union shows complementary detections mainly on clear frames,
not a robust remedy for motion blur. Detection-level bounded persistence is
safer because it leaves current raw pixels and detections unchanged.

## Reproduction

```bash
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
```

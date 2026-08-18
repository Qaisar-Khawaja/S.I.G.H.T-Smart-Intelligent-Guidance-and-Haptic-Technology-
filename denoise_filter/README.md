# S.I.G.H.T. Image Restoration and Tracking Study

This directory contains the complete experimental path for the S.I.G.H.T.
smart-cane computer-vision study: controlled image restoration, evaluation on
real cane-camera footage, pixel-level temporal fusion, filter and temporal
oracles, and detection-level tracking with bounded persistence.

The live cane application is maintained in the parent repository and uses
YOLOv8n. The experimental filters and trackers in this directory were evaluated
offline and are not integrated into that live application.

## Current status

The strongest real-camera result is one-frame ByteTrack persistence:

| Method | Precision | Recall | mAP@0.5 | Rescued | Lost | Stale persisted FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw YOLO11s | 0.8738 | 0.2616 | 0.2591 | 0 | 0 | 0 |
| ByteTrack + 1-frame persistence | 0.8716 | 0.2762 | 0.2698 | 5 | 0 | 1 |

This tracking method is implemented and evaluated in `tracking/evaluate.py`.
It is not integrated into the parent repository's `main.py`; the live
application still uses current-frame YOLOv8n detections.

The retained original-model validation also evaluates YOLOv8n. On Dataset B,
raw YOLOv8n reaches 0.0669 recall and 0.0504 mAP@0.5. BoT-SORT with three-frame
persistence raises these to 0.0727 and 0.0518 by rescuing two clear-frame
instances with no raw losses or stale persisted false positives. It does not
improve moderate- or severe-blur recall. Pixel fusion also underperforms raw
YOLOv8n, while the eight-method union provides no severe-blur recovery; see
`YOLOV8N_VALIDATION.md`.

## Datasets

### Dataset A: controlled synthetic restoration

- 40 clean COCO images in `data/clean/`
- Corresponding YOLO-format labels in `data/clean_labels/`
- Synthetic Gaussian noise, motion blur, low light, and glare
- PSNR, SSIM, detection accuracy, and runtime evaluation

The degradation parameters are pinned in `restoration/degrade.py`. The compared
methods are raw input, Gaussian, bilateral, Wiener denoising, Wiener
deconvolution, and CLAHE.

### Dataset B: real cane-camera benchmark

- 7 original videos in `data/videos/`
- 84 lossless target PNGs in `data/frames_real/`
- 84 human-reviewed YOLO label files in `data/real_labels/`
- 344 scored ground-truth instances
- Quality groups: `clear`, `moderate_blur`, and `severe_blur`
- Frame/video/timestamp metadata in `data/frames_real_metadata.csv`

`data/frames_real/`, `data/real_labels/`, and
`data/frames_real_metadata.csv` are the canonical Dataset B evaluation paths.
The annotation and `smart_cane_frames/` directories retain selection and
annotation provenance.

### Variable-frame-rate warning

`video1.mov` behaves as variable-frame-rate footage. Its stored frame number can
be several decoded frames away from the intended timestamp. Temporal code must
locate targets using decoded `CAP_PROP_POS_MSEC` timestamps, as implemented in
`temporal/neighbors.py`, rather than relying only on
`round(timestamp * fps)`.

## Experimental findings

1. Matched restoration can help on known synthetic degradation in Dataset A.
2. On real Dataset B frames, raw YOLO11s generally beats universal
   single-frame preprocessing.
3. Farneback alignment and pixel fusion degrade detection under large
   cane-sweep ego-motion, especially for severe blur.
4. The filter union raises recall from about 0.262 to 0.329, but most GT objects
   remain missed by every tested preprocessing method.
5. Nearby raw frames provide meaningful temporal headroom: recall reaches
   0.3459 at +/-1, 0.4390 at +/-3, and 0.4826 at +/-5 in the bidirectional
   oracle.
6. The causal past-1 oracle reaches 0.3227 recall.
7. One-frame ByteTrack persistence captures a modest portion of this headroom
   while nearly preserving raw precision.

Detailed reports:

- `YOLOV8N_VALIDATION.md`
- `TEMPORAL_RESTORATION.md`
- `TEMPORAL_ORACLE.md`
- `TRACKING_PERSISTENCE.md`

## Repository layout

```text
.
├── MicroPython/
│   Historical controller copy retained from the experiment branch
├── data/
│   Dataset A, Dataset B, source videos, labels, and metadata
├── annotation/
│   Model-assisted annotation and promotion workflow
├── restoration/
│   Degradations, filters, metrics, and Dataset A/B evaluators
├── temporal/
│   Neighbor lookup, optical flow, quality gates, and pixel fusion
├── tracking/
│   ByteTrack/BoT-SORT configurations and persistence evaluation
├── analysis/
│   Oracles, plots, and qualitative example generation
├── results/
│   CSVs, prediction caches, plots, examples, and tracking states
└── archive/
    Preserved exploratory code and superseded generated examples
```

## Installation

Python 3.11 is recommended.

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For the optional annotation GUI:

```bash
python -m pip install -r requirements-annotation.txt
```

The current environment was validated with NumPy 1.26.4 and Ultralytics
8.4.120. CPU execution is supported; compatible Apple Silicon systems may use
MPS automatically for YOLO inference.

## Model weights

The default evaluator configuration loads `yolo11s.pt`. The live prototype uses
`yolov8n.pt`, and retained validation outputs for that checkpoint are documented
in `YOLOV8N_VALIDATION.md`. Ultralytics may download official weights when
network access is available; otherwise place the required weight file in this
directory before running an evaluator.

Model weights are intentionally ignored by Git. The weight used for the stored
results has this SHA-256 digest:

```text
85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5
```

Verify a local copy on macOS/Linux with:

```bash
shasum -a 256 yolo11s.pt
```

The retained YOLOv8n validation weight has SHA-256 digest:

```text
f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36
```

## Reproducing the experiments

Run commands from this `denoise_filter` directory. The repository tracks
reference outputs, so commands that regenerate them will produce Git changes.
Oracle and tracking commands deliberately require an explicit force option
before overwriting existing outputs.

### Dataset A

```bash
python -m restoration.eval_dataset_a
MPLCONFIGDIR=/private/tmp/sight-mpl-cache python -m analysis.plots
```

Primary output: `results/results_dataset_a.csv`.

To retain a separate original-model run:

```bash
python -m restoration.eval_dataset_a \
  --model yolov8n.pt --confidence 0.25 --seed 0 \
  --output-csv results/yolov8n/results_dataset_a.csv
```

### Dataset B and single-frame filters

```bash
python -m restoration.eval_dataset_b
python -m analysis.hero_examples
```

Primary outputs:

- `results/results_dataset_b_frames.csv`
- `results/results_dataset_b_summary.csv`
- `results/results_dataset_b_groundtruth.csv`
- `results/results_dataset_b_by_quality.csv`
- `results/hero_examples/`

To retain separate YOLOv8n outputs:

```bash
python -m restoration.eval_dataset_b \
  --model yolov8n.pt --confidence 0.15 \
  --output-dir results/yolov8n
```

### Pixel-level temporal restoration

```bash
python -m restoration.eval_temporal
python -m analysis.temporal_hero_examples
MPLCONFIGDIR=/private/tmp/sight-mpl-cache python -m analysis.temporal_plots
```

Primary outputs are `results/temporal_results.csv`,
`results/temporal_frame_analysis.csv`, `results/temporal_examples/`, and the
temporal plots under `results/plots/`.

### Filter oracle

```bash
python -m analysis.oracle_recall
```

Outputs: `results/oracle_union_instances.csv` and
`results/oracle_union_summary.csv`.

### Temporal detection oracle

```bash
python -m analysis.temporal_oracle --force-outputs
python -m analysis.temporal_oracle_causal --force-output
```

Full-video predictions are retained in `results/temporal_oracle_cache/` so
valid caches can be reused without repeating all YOLO inference. Use
`--rebuild-cache` only when the videos, model, threshold, or class set changes.

### Tracking and persistence

```bash
MPLCONFIGDIR=/private/tmp/sight-mpl-cache \
  python -m tracking.evaluate --force-outputs
```

The evaluator requires the seven temporal-oracle cache shards and the causal
oracle summary. It verifies the expected 84 frames, 344 GT instances, and raw
baseline before writing tracking results.

Important outputs include:

- `results/tracking_results.csv`
- `results/tracking_instance_analysis.csv`
- `results/tracking_detection_analysis.csv`
- `results/tracking_oracle_comparison.csv`
- `results/tracking_runtime.csv`
- `results/tracking_frame_states.jsonl`
- `results/tracking_plots/`
- `results/tracking_examples/`

## Annotation workflow

The existing labels are already human-reviewed. To annotate additional selected
frames:

```bash
python -m annotation.select_subset
python -m annotation.prelabel
labelImg annotation/to_label
python -m annotation.promote_labels
```

`prelabel` creates drafts only. Every draft must be manually corrected before
promotion into `data/real_labels/`.

## Live application

The live entry point remains in the parent repository:

```bash
cd ..
python main.py
```

It expects the configured camera and serial-connected Pico. Camera index and
serial-port settings are currently defined in the existing live source files.
Press `q` in the OpenCV window to exit. Hardware source and configuration should
be validated on the actual demonstration system before changing them.

## Evaluation notes

- YOLO confidence is 0.15 for the real-camera evaluation paths.
- Relevant classes are defined centrally in `restoration/classes.py`.
- Detection matching uses same-class IoU >= 0.5.
- `restoration/detection_metrics.py` is the shared mAP implementation.
- The stored raw mAP@0.5 and mAP@0.5:0.95 are both 0.2591. The stricter sweep
  was implemented separately over IoU thresholds 0.50 through 0.95; the unusual
  equality is retained for later verification rather than altered during
  repository cleanup.
- Nearby video frames are correlated, so the 84 targets should not be described
  as statistically independent samples.

## Reproducibility policy

Datasets, annotations, metadata, result tables, plots, qualitative failures, and
expensive full-video prediction caches are intentionally tracked. Virtual
environments, Python caches, local agent/editor settings, and model weights are
ignored. Do not remove duplicate-looking annotation or dataset exports without
first confirming their provenance and canonical consumer paths.

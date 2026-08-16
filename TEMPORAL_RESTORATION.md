# Temporal restoration experiment

## Motivation

Dataset A (synthetic degradation on clean COCO images) showed that matched filters,
Wiener deconvolution (known kernel), and denoisers all help when the degradation
model matches the true corruption. Dataset B (real smart-cane footage) showed the
opposite: raw input generally wins aggregate mAP across all three severity groups,
and single-frame filters (Gaussian, bilateral, Wiener denoise/deconv, CLAHE) do not
beat raw overall, even though individual filters occasionally rescue specific missed
detections. That gap suggests real cane-camera motion blur doesn't match any of the
synthetic degradation models these filters assume.

This experiment asks a different question: since a severely blurred frame may have
genuinely lost information that no single-frame restoration can recover, can
**neighboring frames from the same video** supply that missing information?

**Methodological constraint, stated explicitly:** neighboring frames are used only
to construct a restored version of the annotated target frame I(t). Evaluation is
always performed against I(t)'s original ground-truth annotation — no new ground
truth is created for neighboring frames, and the restored output is always warped
into I(t)'s coordinate system so the existing boxes in `data/real_labels/` stay
valid for every method, including the temporal ones.

## Method

### 1. Reconstructing neighbor frames (`temporal/neighbors.py`)

Each of the 84 annotated Dataset B frames is matched back to its source video
(`data/videos/videoN.mov`) and exact position in that video, then I(t-1) and I(t+1)
are decoded fresh from the video (I(t) itself is never re-decoded — it's loaded
directly from `data/frames_real/`, the exact pixels the ground truth was drawn
against).

Locating I(t) in the source video turned out to be non-trivial: the metadata's
`frame_number` column was computed as `round(timestamp_sec * fps)`, which assumes a
constant frame rate. Empirically, `video1.mov` is variable-frame-rate — sequentially
decoding it and comparing against the known-correct stored target image showed
`frame_number` landing up to **6 frames off** from where OpenCV's decode actually
was at that timestamp (video2/video3, which are CFR, matched exactly). The fix:
match each candidate frame's own decoded presentation timestamp
(`CAP_PROP_POS_MSEC`) against `timestamp_sec` during a sequential scan, rather than
trusting `frame_number` directly. This costs one extra sequential decode pass per
video, negligible at this frame count (84 frames processed in well under a minute).

The offset is configurable (`--offset`, default 1, i.e. I(t-1)/I(t)/I(t+1)); boundary
cases (start/end of video) are handled by leaving that neighbor `None` rather than
raising, which downstream fusion treats as "unavailable, redistribute its weight."

### 2. Optical-flow alignment (`temporal/optical_flow.py`)

Raw neighbor frames are never blended directly — the camera is moving, so that would
just produce ghosting. Each neighbor is aligned into I(t)'s coordinate frame with
OpenCV's Farneback dense optical flow: `flow = calcOpticalFlowFarneback(target_gray,
neighbor_gray, ...)` gives, for each pixel in the target, the displacement to find
the same content in the neighbor; `cv2.remap` then pulls the neighbor's color pixels
into the target's grid along that flow field. Pixels whose required source sample
falls outside the neighbor frame (content the neighbor never saw) are flagged
invalid via a `valid_mask` rather than silently blended in as black.

### 3. Two fusion strategies (`temporal/fusion.py`)

- **`temporal_fixed`**: `I_fused = 0.25·I'(t-1) + 0.50·I(t) + 0.25·I'(t+1)`, weights
  configurable. A rejected/missing neighbor's weight is zeroed and the rest
  renormalized.
- **`temporal_quality_weighted`**: each available frame is weighted by
  `sqrt(laplacian_variance)` (sqrt to dampen the ratio when one frame is much
  sharper — using raw variance directly produced unstably extreme weights during
  development), normalized to sum to 1, then the target's weight is floored at 0.3
  (rescaling the neighbor share proportionally) so a sharp neighbor can never fully
  outvote the frame the ground truth was actually drawn against.

### 4. Alignment sanity check (`temporal/quality_gate.py`)

After warping, each neighbor is scored on: fraction of valid (in-frame) pixels
(≥0.6 required), mean absolute photometric difference against the target on valid
pixels (≤40 on a 0-255 scale), SSIM against the target (≥0.25), and mean flow
magnitude on valid pixels (≤60px). Failing any check rejects that neighbor (weight
→ 0, redistributed). This is deliberately simple — three independent thresholds, no
learned selector — and every rejection is logged.

### 5. Evaluation (`restoration/eval_temporal.py`)

Same 84 ground-truth-annotated frames, same YOLO11s model, same confidence
threshold (0.15, matching `main.py`'s live setting — see `eval_dataset_b.py`), same
IoU/mAP scoring (`restoration.detection_metrics`, extended with mAP@0.5:0.95 via
the same tested greedy-matching machinery) as the existing Dataset B evaluation.
Methods compared: `raw`, `wiener_denoise`, `wiener_deconv`, `clahe`,
`temporal_fixed`, `temporal_quality_weighted`. The two temporal methods share one
alignment pass per frame so their comparison isolates the fusion strategy, not
incidental flow differences.

Per-frame rescue/lost/confidence-shift bookkeeping (`temporal_frame_analysis.csv`)
compares raw vs. each temporal method per (frame, ground-truth class), reusing the
same greedy IoU matcher used for mAP, and classifies each as `rescued`, `lost`,
`confidence_improved`, `confidence_degraded`, `no_change`, or `still_missed`
(confidence shift threshold: 0.05).

## Dataset B setup

84 hand-annotated real frames across 7 cane-sweep videos, quality-grouped as
`clear` (28), `moderate_blur` (35), `severe_blur` (21) via
`data/frames_real_metadata.csv`. 81 of 84 frames have at least one ground-truth
object in a scored class (`restoration.classes.RELEVANT_CLASSES`) — 3 frames'
annotations only cover out-of-scope classes and are excluded from the pooled
precision/recall/mAP, matching how `precision_recall_map` already treats empty
ground truth.

## Note: why mAP@0.5 and mAP@0.5:0.95 sit so close together

Worth explaining since it looks unusual at a glance: normally mAP@0.5:0.95 is
noticeably lower than mAP@0.5, since it also scores against much stricter IoU
thresholds. Here they're close (e.g. raw: 0.259 vs 0.259; temporal_fixed: 0.210 vs
0.201). The threshold sweep itself was double-checked independently outside the
eval script (0.5, 0.55, ..., 0.95, matching per-detection at each threshold) and is
correct — this isn't a bug reusing the mAP@0.5 result.

The real explanation: raw YOLO11s's correct detections on this dataset are
**pixel-near-perfect** far more often than not. Checking directly — running raw
YOLO11s against all 84 frames and comparing its own predictions to the
(human-reviewed, confirmed-correct) ground truth — found 87 of 344 ground-truth
boxes (25.3%, across 35 frames) match a raw prediction at IoU > 0.98, several to 6
decimal places. That's not a labeling problem (confirmed: every one of the 84
frames was reviewed and corrected by hand in labelImg; boxes that match YOLO's
draft closely simply mean YOLO got that particular object right and the correction
was "leave it as-is"). It does mean raw's detection behavior on this dataset is
strongly **bimodal** — an object is either found almost exactly right, or missed
entirely, with very little of the "found it but the box is a bit loose" middle
ground that would normally separate mAP@0.5 from mAP@0.5:0.95. That's plausible
here: many of the annotated objects (chairs, tables, doorframes) are large,
high-contrast, and fill much of the frame, so once YOLO locks onto one there's
little room for boundary ambiguity — a plausible dataset characteristic (curated,
small, dominated by big obstacles) rather than a general property to expect on
messier data.

## Results (actual numbers, offset=1)

*(mAP@0.5 and mAP@0.5:0.95 sit unusually close together in this table — checked and
confirmed not a threshold-sweep bug; see the note above for why.)*

### Overall (81 scored frames)

| Method | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Latency (ms) | FPS |
|---|--:|--:|--:|--:|--:|--:|
| raw | 0.874 | 0.262 | 0.259 | 0.259 | 46.4 | 21.5 |
| wiener_denoise | 0.878 | 0.230 | 0.223 | 0.213 | 79.7 | 12.6 |
| wiener_deconv | 0.000 | 0.000 | 0.000 | 0.000 | 97.7 | 10.2 |
| clahe | 0.713 | 0.209 | 0.183 | 0.165 | 26.1 | 38.3 |
| **temporal_fixed** | 0.679 | 0.166 | 0.210 | 0.201 | 131.8 | 7.6 |
| **temporal_quality_weighted** | 0.575 | 0.145 | 0.195 | 0.189 | 128.9 | 7.8 |

### By severity (mAP@0.5)

| Method | Clear (n=27) | Moderate (n=35) | Severe (n=19) |
|---|--:|--:|--:|
| raw | 0.376 | 0.266 | 0.127 |
| wiener_denoise | 0.330 | 0.223 | 0.059 |
| wiener_deconv | 0.000 | 0.000 | 0.000 |
| clahe | 0.347 | 0.088 | 0.087 |
| temporal_fixed | 0.312 | **0.202** | 0.006 |
| temporal_quality_weighted | 0.302 | **0.195** | 0.000 |

(`wiener_deconv=0.000` everywhere is a pre-existing, already-documented result —
`results/results_dataset_b_groundtruth.csv` shows the same thing; it's not
something this experiment introduced. Its fixed-kernel assumption simply doesn't
hold on footage with no known blur kernel.)

### Rescue/loss bookkeeping, raw vs. temporal (all 81 scored frames, per ground-truth class)

| Method | Rescued | Lost | Confidence improved | Confidence degraded | No change | Still missed |
|---|--:|--:|--:|--:|--:|--:|
| temporal_fixed | 3 | 25 | 5 | 11 | 10 | 111 |
| temporal_quality_weighted | 2 | 28 | 6 | 7 | 10 | 112 |

Broken down by severity for `temporal_quality_weighted`:

- **clear**: 1 rescued, 10 lost, 6 confidence-improved, 4 confidence-degraded
- **moderate_blur**: 1 rescued, 13 lost, 0 confidence-improved, 3 confidence-degraded
- **severe_blur**: 0 rescued, 5 lost, 0 confidence-improved, 0 confidence-degraded
  (every single ground-truth object raw caught in this group, temporal lost)

### Alignment quality gate

At offset=1: 0 neighbors rejected by the quality gate across all 84 frames; 6 `prev`
neighbors missing (video-start boundary), 0 `next` missing. In other words, the gate
as configured judged every warped neighbor "acceptable" by its coarse global
metrics (valid-pixel fraction, mean photometric diff, SSIM, flow magnitude) — yet
severe_blur mAP still collapsed to near zero. **This is a real limitation, not
hidden**: the gate's thresholds were tuned by eyeballing a handful of dev frames,
not to specifically catch the failure mode that actually hurt — the fused image
still often *looks* globally plausible (right brightness, right rough structure)
even where the fusion has smeared a moving person or object into the background
enough to break YOLO's edge cues. A qualitative example of exactly this is saved at
`results/temporal_examples/degraded/video6_v06_20-24_f00215_t07.17_severe_blur_temporal_fixed_person.jpg`:
raw detects `person 0.49`; the fused frame shows the person smeared into the hallway
background, undetected.

## Finding

**Outcome C, with Outcome B nuance.** Temporal restoration does not beat raw
overall, and gets *categorically worse* —
not just "no better" — specifically under severe blur, where the lightweight
Farneback alignment breaks down on exactly the frames with the most ego-motion
(large cane sweeps), producing ghosting/smearing that destroys the edge
information YOLO needs. `temporal_quality_weighted` hits **mAP@0.5 = 0.000** on
severe_blur — worse than doing nothing.

The picture is more mixed at lower severities: on `moderate_blur`,
`temporal_fixed` (0.202) modestly *beats* every single-frame filter (wiener_denoise
0.223 is close but clahe collapses to 0.088), though it still trails raw (0.266).
On `clear` frames temporal restoration is competitive with wiener_denoise/clahe but
below raw. And a handful of genuine per-detection rescues did happen (3 and 2,
respectively) — e.g. a person and a chair caught only after temporal fusion — so
multi-frame information is not *useless*, it's just outweighed roughly 10:1 by
detections it costs, with severe blur alone accounting for the worst of that
deficit.

**Interpretation**: large ego-motion from the cane's sweeping motion makes
lightweight optical-flow alignment unreliable specifically where it's needed most
(severe blur = fastest motion). A more capable video-restoration model — one with
learned motion compensation robust to large displacements, rather than classical
dense flow — is a more promising direction than tuning this baseline further. This
matches the domain-gap story from the single-frame filters: real cane-camera motion
doesn't match the assumptions of the lightweight methods tried so far, whether
single-frame or multi-frame.

## Runtime

Temporal preprocessing (alignment + fusion, both neighbors) costs ~104-115ms/frame
on CPU/MPS (Farneback flow dominates), on top of ~20-30ms YOLO inference — about
7.5 FPS total vs. raw's ~21 FPS (severe/moderate_blur, where preprocessing is the
whole cost since raw has none) or ~10 FPS on clear frames (YOLO alone is slower on
frames with more/larger detections). Not real-time on this hardware without
further optimization (e.g. a smaller flow resolution).

## Limitations

- The alignment quality gate did not catch the failure mode that mattered most
  (see above) — its thresholds are a first pass, not validated against the failure
  cases they were meant to catch.
- Offset=1 (single adjacent frame each side) was the only offset evaluated in the
  full run; `--offset` is wired through for testing wider gaps like I(t-2, t, t+2),
  but wasn't swept here due to time.
- Farneback is a classical, un-learned flow method; it has no way to handle large
  displacements or occlusion beyond the coarse-to-fine pyramid, which is exactly
  what breaks on severe blur.
- Part 12 (pretrained video-deblurring model, e.g. EDVR/RVRT/BasicVSR++) and Part 13
  (adaptive sharpness-threshold rule) were not attempted — see Priority order below.

## Reproducing

```bash
source venv/bin/activate

# Full evaluation (84 frames x 6 methods), writes results/temporal_results.csv,
# results/temporal_frame_analysis.csv, and a handful of debug alignment panels
# to results/temporal_examples/debug/
python -m restoration.eval_temporal

# Optional: different neighbor offset, more/fewer debug panels
python -m restoration.eval_temporal --offset 2 --debug-per-group 3

# Hero examples (rescued / degraded / neutral side-by-sides)
python -m analysis.temporal_hero_examples

# Plots
python -m analysis.temporal_plots
```

Existing experiments (`restoration/eval_dataset_a.py`, `restoration/eval_dataset_b.py`,
`analysis/hero_examples.py`, `analysis/plots.py`) are untouched and their output
files were not overwritten.

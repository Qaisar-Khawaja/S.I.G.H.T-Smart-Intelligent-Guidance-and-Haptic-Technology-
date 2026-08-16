# Archive

This directory preserves material that is no longer authoritative but remains
useful for research history. Nothing here should be treated as the current
evaluation or live-demo entry point.

## Contents

- `exploratory/restoration/sample_frames_25_per_video.py`: an earlier
  sharpness-bucket sampler that selected approximately 25 frames per video. It
  predates the current manually curated 84-frame Dataset B benchmark and must
  not be used to regenerate `data/frames_real/`.
- `superseded_results/tracking_examples/`: BoT-SORT persistence example strips
  retained from runs where a different configuration was selected for example
  generation. The complete BoT-SORT metrics and ablations remain in the active
  tracking CSVs; the current report-selected examples use one-frame ByteTrack
  persistence.

Canonical experiment entry points and result locations are documented in the
repository root `README.md`.

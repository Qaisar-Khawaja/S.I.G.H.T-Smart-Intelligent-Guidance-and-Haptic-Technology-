# Detection-Level Tracking and Persistence

## Research question

Can causal object-state tracking bridge temporary raw-YOLO misses in cane-sweep video without creating unacceptable stale false positives? This follows the negative pixel-restoration result and the positive temporal-oracle headroom analysis.

## Causal oracle sanity check

| Group | Raw | Past 1 | Past 3 | Past 5 | Future 5 | ±5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 0.0669 | 0.0814 | 0.1017 | 0.1134 | 0.1308 | 0.1715 |
| clear | 0.1120 | 0.1280 | 0.1680 | 0.1840 | 0.2320 | 0.2880 |
| moderate_blur | 0.0519 | 0.0714 | 0.0779 | 0.0909 | 0.0974 | 0.1364 |
| severe_blur | 0.0154 | 0.0154 | 0.0308 | 0.0308 | 0.0154 | 0.0308 |

Past-only headroom remains meaningful overall and for moderate blur. Severe blur has less causal headroom, and the bidirectional ceiling would require future-frame buffering.

## Tracking method

ByteTrack and BoT-SORT are the implementations bundled with Ultralytics 8.4.120. Their default thresholds are pinned in `tracking/*_sight.yaml`: high/new-track 0.25, low 0.10, match 0.8, and lost-track buffer 30. YOLO input is `yolov8n.pt` at confidence 0.15. BoT-SORT uses built-in sparse-optical-flow global motion compensation and no ReID.

Standard tracker output omits unconfirmed and unmatched tracks. For safety, persistence configurations pass every current raw YOLO observation through unchanged and add unmatched Kalman-predicted states for at most 1, 3, or 5 frames. At annotated targets, the pass-through observations come from the exact target PNG used by the established raw baseline; only genuine predicted states come from continuous video tracking, preventing variable-frame-rate decode differences from masquerading as rescues/losses. Predicted confidence = last detection confidence × 0.9^missed_frames; boxes are clipped and decayed confidence below 0.15 is dropped. A predicted lost box overlapping a current same-class observation at IoU >= 0.5 is suppressed as a duplicate.

## Results

| Method | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | TP | FP | Rescued | Lost | Stale persisted FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | 0.7667 | 0.0669 | 0.0504 | 0.0467 | 23 | 7 | 0 | 0 | 0 |
| bytetrack | 1.0000 | 0.0378 | 0.0244 | 0.0201 | 13 | 0 | 0 | 10 | 0 |
| botsort | 1.0000 | 0.0407 | 0.0251 | 0.0233 | 14 | 0 | 0 | 9 | 0 |
| botsort_persist_3 | 0.7812 | 0.0727 | 0.0518 | 0.0477 | 25 | 7 | 2 | 0 | 0 |

### Recall by severity

| Method | Clear | Moderate blur | Severe blur | Overall |
| --- | ---: | ---: | ---: | ---: |
| raw | 0.1120 | 0.0519 | 0.0154 | 0.0669 |
| bytetrack | 0.0720 | 0.0260 | 0.0000 | 0.0378 |
| botsort | 0.0800 | 0.0260 | 0.0000 | 0.0407 |
| botsort_persist_3 | 0.1280 | 0.0519 | 0.0154 | 0.0727 |

### Persistence trade-off

| Method | Rescued | Lost | Stale FP | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| bytetrack_persist_1 | 1 | 0 | 0 | 0.7742 | 0.0698 |
| bytetrack_persist_3 | 2 | 0 | 1 | 0.7576 | 0.0727 |
| bytetrack_persist_5 | 2 | 0 | 2 | 0.7353 | 0.0727 |
| botsort_persist_1 | 1 | 0 | 0 | 0.7742 | 0.0698 |
| botsort_persist_3 | 2 | 0 | 0 | 0.7812 | 0.0727 |
| botsort_persist_5 | 2 | 0 | 1 | 0.7576 | 0.0727 |

The selected trade-off is `botsort_persist_3`: configurations within 0.005 of the best overall F1 are tie-broken toward fewer stale persisted false positives, then higher precision.

## Oracle exploitation

For botsort_persist_3, actual overall recall is 0.0727 versus a past-3 oracle of 0.1017. It recovers 16.7% of available causal recall headroom.

| Group | Raw | Past oracle | Actual | Headroom recovered |
| --- | ---: | ---: | ---: | ---: |
| overall | 0.0669 | 0.1017 | 0.0727 | 16.7% |
| clear | 0.1120 | 0.1680 | 0.1280 | 28.6% |
| moderate_blur | 0.0519 | 0.0779 | 0.0519 | 0.0% |
| severe_blur | 0.0154 | 0.0308 | 0.0154 | 0.0% |

## Runtime

| Method | YOLO wall ms | Tracking ms | Total ms | FPS |
| --- | ---: | ---: | ---: | ---: |
| raw | 52.697 | 0.000 | 52.697 | 18.98 |
| bytetrack | 52.697 | 0.638 | 53.335 | 18.75 |
| botsort | 52.697 | 6.500 | 59.197 | 16.89 |
| botsort_persist_3 | 52.697 | 6.506 | 59.203 | 16.89 |

The runtime benchmark is a sequential single-frame measurement on the current CPU environment. Earlier stored YOLO11s measurements were approximately 21.5 FPS for raw and 7.6–7.8 FPS for pixel-level temporal restoration; hardware/runtime state can make direct absolute comparisons noisy.

## Pixel state versus object state

Farneback warping plus pixel fusion previously degraded severe-blur YOLO11s mAP@0.5 from 0.127 to 0.006/0.000. Detection-level persistence leaves raw pixels untouched and instead carries a bounded, decaying tracker state. The current tracking result determines whether that distinction is practically useful rather than merely theoretically promising.

## Examples and failure modes

Best successful strip: `results/yolov8n/tracking_examples/rescued/video1_v01_20-16_f00325_t10.94_clear_botsort_persist_3_rescued_gt000.jpg`

Worst stale-failure strip: `none`

Green boxes are raw YOLO, magenta boxes are tracker/persistent state, and red target-frame boxes are GT. Lost and stale-false-positive folders are included alongside successful and neutral examples to avoid success-only selection.
Because the selected safety-preserving persistence wrapper loses no raw GT detections, its `lost/` examples show the standard BoT-SORT baseline suppressing current detections rather than a persistence loss.

## Reproduction

```bash
venv/bin/pip install -r requirements.txt
venv/bin/python -m analysis.temporal_oracle --model yolov8n.pt --confidence 0.15 --output-dir results/yolov8n --report results/yolov8n/TEMPORAL_ORACLE.md --force-outputs
venv/bin/python -m analysis.temporal_oracle_causal --instances-csv results/yolov8n/temporal_oracle_instances.csv --output-csv results/yolov8n/temporal_oracle_causal_summary.csv --force-output
MPLCONFIGDIR=/private/tmp/sight-mpl-cache venv/bin/python -m tracking.evaluate --model yolov8n.pt --confidence 0.15 --cache-dir results/yolov8n/temporal_oracle_cache --causal-oracle results/yolov8n/temporal_oracle_causal_summary.csv --output-dir results/yolov8n --report results/yolov8n/TRACKING_PERSISTENCE.md --force-outputs
```

## Conclusion

Detection persistence provides a modest positive result: it bridges a small number of short misses while essentially preserving raw precision, but remains far below the causal oracle under ego-motion.

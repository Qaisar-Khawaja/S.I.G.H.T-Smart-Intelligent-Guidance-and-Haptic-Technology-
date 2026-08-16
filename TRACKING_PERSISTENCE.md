# Detection-Level Tracking and Persistence

## Research question

Can causal object-state tracking bridge temporary raw-YOLO misses in cane-sweep video without creating unacceptable stale false positives? This follows the negative pixel-restoration result and the positive temporal-oracle headroom analysis.

## Causal oracle sanity check

| Group | Raw | Past 1 | Past 3 | Past 5 | Future 5 | ±5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 0.2616 | 0.3227 | 0.3895 | 0.4186 | 0.3808 | 0.4826 |
| clear | 0.3760 | 0.4400 | 0.5040 | 0.5280 | 0.4640 | 0.5600 |
| moderate_blur | 0.2403 | 0.3117 | 0.4026 | 0.4416 | 0.3961 | 0.5260 |
| severe_blur | 0.0923 | 0.1231 | 0.1385 | 0.1538 | 0.1846 | 0.2308 |

Past-only headroom remains meaningful overall and for moderate blur. Severe blur has less causal headroom, and the bidirectional ceiling would require future-frame buffering.

## Tracking method

ByteTrack and BoT-SORT are the implementations bundled with Ultralytics 8.4.120. Their default thresholds are pinned in `tracking/*_sight.yaml`: high/new-track 0.25, low 0.10, match 0.8, and lost-track buffer 30. YOLO input remains `yolo11s.pt` at confidence 0.15. BoT-SORT uses built-in sparse-optical-flow global motion compensation and no ReID.

Standard tracker output omits unconfirmed and unmatched tracks. For safety, persistence configurations pass every current raw YOLO observation through unchanged and add unmatched Kalman-predicted states for at most 1, 3, or 5 frames. At annotated targets, the pass-through observations come from the exact target PNG used by the established raw baseline; only genuine predicted states come from continuous video tracking, preventing variable-frame-rate decode differences from masquerading as rescues/losses. Predicted confidence = last detection confidence × 0.9^missed_frames; boxes are clipped and decayed confidence below 0.15 is dropped. A predicted lost box overlapping a current same-class observation at IoU >= 0.5 is suppressed as a duplicate.

## Results

| Method | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | TP | FP | Rescued | Lost | Stale persisted FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | 0.8738 | 0.2616 | 0.2591 | 0.2591 | 90 | 13 | 0 | 0 | 0 |
| bytetrack | 0.9153 | 0.1570 | 0.1885 | 0.1723 | 54 | 5 | 4 | 40 | 0 |
| botsort | 0.8906 | 0.1657 | 0.1883 | 0.1814 | 57 | 7 | 3 | 36 | 0 |
| bytetrack_persist_1 | 0.8716 | 0.2762 | 0.2698 | 0.2634 | 95 | 14 | 5 | 0 | 1 |

### Recall by severity

| Method | Clear | Moderate blur | Severe blur | Overall |
| --- | ---: | ---: | ---: | ---: |
| raw | 0.3760 | 0.2403 | 0.0923 | 0.2616 |
| bytetrack | 0.2240 | 0.1494 | 0.0462 | 0.1570 |
| botsort | 0.2320 | 0.1558 | 0.0615 | 0.1657 |
| bytetrack_persist_1 | 0.4000 | 0.2468 | 0.1077 | 0.2762 |

### Persistence trade-off

| Method | Rescued | Lost | Stale FP | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| bytetrack_persist_1 | 5 | 0 | 1 | 0.8716 | 0.2762 |
| bytetrack_persist_3 | 7 | 0 | 9 | 0.8151 | 0.2820 |
| bytetrack_persist_5 | 7 | 0 | 13 | 0.7886 | 0.2820 |
| botsort_persist_1 | 5 | 0 | 1 | 0.8716 | 0.2762 |
| botsort_persist_3 | 6 | 0 | 3 | 0.8571 | 0.2791 |
| botsort_persist_5 | 6 | 0 | 8 | 0.8205 | 0.2791 |

The selected trade-off is `bytetrack_persist_1`: configurations within 0.005 of the best overall F1 are tie-broken toward fewer stale persisted false positives, then higher precision.

## Oracle exploitation

For bytetrack_persist_1, actual overall recall is 0.2762 versus a past-1 oracle of 0.3227. It recovers 23.9% of available causal recall headroom.

| Group | Raw | Past oracle | Actual | Headroom recovered |
| --- | ---: | ---: | ---: | ---: |
| overall | 0.2616 | 0.3227 | 0.2762 | 23.9% |
| clear | 0.3760 | 0.4400 | 0.4000 | 37.5% |
| moderate_blur | 0.2403 | 0.3117 | 0.2468 | 9.1% |
| severe_blur | 0.0923 | 0.1231 | 0.1077 | 50.0% |

## Runtime

| Method | YOLO wall ms | Tracking ms | Total ms | FPS |
| --- | ---: | ---: | ---: | ---: |
| raw | 129.009 | 0.000 | 129.009 | 7.75 |
| bytetrack | 129.009 | 1.012 | 130.021 | 7.69 |
| botsort | 129.009 | 9.518 | 138.527 | 7.22 |
| bytetrack_persist_1 | 129.009 | 1.030 | 130.039 | 7.69 |

The runtime benchmark is a sequential single-frame measurement on the current CPU environment. Existing stored project measurements were approximately 21.5 FPS for raw and 7.6–7.8 FPS for pixel-level temporal restoration; hardware/runtime state can make direct absolute comparisons noisy.

## Pixel state versus object state

Farneback warping plus pixel fusion degraded severe-blur mAP@0.5 from 0.127 to 0.006/0.000. Detection-level persistence leaves raw pixels untouched and instead carries a bounded, decaying tracker state. The tracking result below determines whether that distinction is practically useful rather than merely theoretically promising.

## Examples and failure modes

Best successful strip: `results/tracking_examples/rescued/video3_v03_20-20_f00113_t03.77_severe_blur_bytetrack_persist_1_rescued_gt000.jpg`

Worst stale-failure strip: `results/tracking_examples/stale_false_positive/video1_v01_20-16_f00325_t10.94_clear_bytetrack_persist_1_stale_false_positive.jpg`

Green boxes are raw YOLO, magenta boxes are tracker/persistent state, and red target-frame boxes are GT. Lost and stale-false-positive folders are included alongside successful and neutral examples to avoid success-only selection.
Because the selected safety-preserving persistence wrapper loses no raw GT detections, its `lost/` examples show the standard ByteTrack baseline suppressing current detections rather than a persistence loss.

## Reproduction

```bash
venv/bin/pip install -r requirements.txt
venv/bin/python -m analysis.temporal_oracle_causal --force-output
MPLCONFIGDIR=/private/tmp/sight-mpl-cache venv/bin/python -m tracking.evaluate --force-outputs
```

## Conclusion

Detection persistence provides a modest positive result: it bridges a small number of short misses while essentially preserving raw precision, but remains far below the causal oracle under ego-motion.

# SmartCane: A Vision-Guided Navigation Aid for Visually Impaired Users

**Khawaja Faiza Qaisar – 217948233**
**Tran Tran – 220168829**
**Richard Balroop – 216906349**

## Overview

SmartCane is an assistive navigation prototype designed to help visually impaired users detect obstacles before physical contact occurs.

Traditional white canes primarily detect obstacles at ground level through physical contact. SmartCane extends this capability by combining **real-time computer vision, object detection, haptic feedback, and directional audio alerts**.

A camera mounted on the cane captures the user's surroundings. A host computer runs a lightweight **YOLOv8n object detector** and determines:

* What obstacle has been detected
* Whether the obstacle is on the left, center, or right
* Whether the obstacle is far, medium, or close
* Which detected obstacle represents the most urgent hazard

The resulting hazard state is transmitted wirelessly over **UDP/WiFi** to a **Raspberry Pi Pico**, which controls a vibration motor and a DFPlayer Mini audio module.

The project also includes a systematic evaluation of image restoration and temporal-processing techniques for motion-blurred cane-camera footage. These experiments showed that restoration methods that improve detection under controlled synthetic degradation generally do not transfer to real cane-mounted footage. Short-term object-level temporal persistence was instead found to be a more promising approach.

---

## System Architecture

```text
                    Camera
                       |
                       v
              +------------------+
              |  Host Computer   |
              |                  |
              |  YOLOv8n         |
              |      |           |
              |      v           |
              | Hazard Detection |
              |      |           |
              |      v           |
              |   Hysteresis     |
              |      |           |
              |      v           |
              | UDP/WiFi Command |
              +--------+---------+
                       |
                    WiFi / UDP
                       |
                       v
              +------------------+
              | Raspberry Pi Pico|
              |                  |
              | MicroPython      |
              | UDP Watchdog     |
              +--------+---------+
                       |
             +---------+---------+
             |                   |
             v                   v
       Vibration Motor      DFPlayer Mini
                                 |
                                 v
                              Speaker
```

### Processing Pipeline

```text
Camera
  ↓
YOLOv8n Detection
  ↓
Relevant Object Filtering
  ↓
Distance + Direction Estimation
  ↓
Hazard Scoring
  ↓
Highest-Priority Hazard
  ↓
Temporal Hysteresis
  ↓
Two-Character Command
  ↓
UDP / WiFi
  ↓
Raspberry Pi Pico
  ↓
 ┌───────────────┬────────────────┐
 │               │                │
 Vibration      DFPlayer Mini    Watchdog
 Motor           + Speaker
```

---

## Key Features

* Real-time object detection using YOLOv8n
* Detection of multiple obstacles in a single frame
* Left / center / right hazard classification
* Distance approximation using bounding-box height
* Priority-based hazard selection
* Three urgency levels
* Directional audio feedback
* Three-level haptic feedback
* Wireless UDP communication between host and Raspberry Pi Pico
* Audio output through a DFPlayer Mini and speaker
* Command hysteresis for temporal stabilization
* One-second command retransmission
* Three-second microcontroller link watchdog
* Automatic motor shutdown when communication is lost
* Evaluation of image restoration techniques for motion-blurred cane footage
* Evaluation of pixel-level temporal fusion
* Evaluation of object-level temporal persistence using ByteTrack and BoT-SORT
* Comparison between YOLOv8n and YOLO11s on real cane-camera footage

---

# Hardware

## Prototype Hardware

* Raspberry Pi Pico
* Camera / webcam
* Vibration motor
* DFPlayer Mini MP3 module
* 8 Ω, 2 W speaker
* microSD card
* Cane mounting hardware
* Host computer/laptop
* WiFi network

The host computer performs computationally intensive object detection and hazard reasoning. The Raspberry Pi Pico acts as the real-time hardware controller responsible for vibration, audio playback, network monitoring, and safe-state handling.

---

## Raspberry Pi Pico

The Raspberry Pi Pico runs **MicroPython** and performs:

* UDP packet reception
* Hazard command interpretation
* Vibration motor control
* DFPlayer Mini communication
* Audio track selection
* Link watchdog monitoring
* Automatic safe-state recovery when communication is lost

The MicroPython firmware is deployed directly to the Pico's onboard flash memory using **Thonny IDE**.

The system can therefore operate from USB power-bank delivery without requiring terminal intervention on the Pico.

---

## DFPlayer Mini Audio Module

Directional feedback is provided through a **DFPlayer Mini MP3 module** connected to an 8 Ω, 2 W speaker.

The DFPlayer was selected because it contains its own:

* Hardware MP3 decoder
* Storage interface
* Audio amplifier

This allows the Pico to issue simple track-selection commands instead of performing real-time audio processing.

### Interface

The DFPlayer communicates with the Raspberry Pi Pico over a **9600-baud UART connection**.

The module uses fixed ten-byte command frames:

```text
0x7E 0xFF 0x06 [CMD] 0x00
[PARAM HI] [PARAM LO]
[CHK HI] [CHK LO] 0xEF
```

The firmware encapsulates frame construction so that the rest of the application only needs to specify the desired track.

### Audio Tracks

The microSD card contains three directional audio clips:

```text
0001.mp3 → "left"
0002.mp3 → "right"
0003.mp3 → "ahead"
```

The clips were generated using the host operating system's text-to-speech engine.

Because some DFPlayer-compatible modules index tracks according to file copy order rather than filename, the files are copied individually and sequentially to the microSD card.

---

# Software Requirements

## Host Computer

The host application requires Python 3 and the project's computer-vision and machine-learning dependencies.

Major components include:

* Python 3
* Ultralytics YOLO
* YOLOv8n
* OpenCV
* NumPy
* PyTorch
* UDP socket communication

Additional packages are required for the restoration, optical-flow, and tracking experiments.

See [`denoise_filter/requirements.txt`](denoise_filter/requirements.txt) for the experimental environment.

## Raspberry Pi Pico

The Pico requires:

* MicroPython
* UDP/network support
* UART support
* GPIO support

The firmware can be installed and managed using **Thonny IDE**.

---

# Hazard Detection

SmartCane processes every YOLO detection and converts it into a hazard state based on its apparent distance and horizontal position.

## Object Filtering

The detector is restricted to relevant obstacle classes, including examples such as:

* Bottles
* Cellphones
* Bags
* Chairs
* People

The set of relevant classes can be modified according to the deployment requirements.

---

## Direction

The camera image is divided into three vertical regions.

```text
+----------------+----------------+----------------+
|                |                |                |
|      LEFT      |     CENTER     |     RIGHT      |
|                |                |                |
+----------------+----------------+----------------+
```

For a frame with width `W`, the horizontal center of the bounding box is:

```text
xcenter = (x1 + x2) / 2
```

The direction is determined as:

```text
Left:
xcenter < W/3

Center:
W/3 ≤ xcenter ≤ 2W/3

Right:
xcenter > 2W/3
```

---

## Distance Approximation

SmartCane uses bounding-box height as an approximate distance indicator.

```text
h = y2 - y1
```

The current thresholds are:

| Distance | Bounding-box height |
| -------- | ------------------: |
| Close    |            > 250 px |
| Medium   |          120–250 px |
| Far      |            ≤ 120 px |

Far objects are ignored by the hazard-scoring system.

This is a **relative distance proxy**, not a calibrated physical distance measurement.

---

# Hazard Scoring

Every relevant detection receives an urgency score:

```text
Hazard Score = Distance Score + Direction Bonus
```

| Condition | Score |
| --------- | ----: |
| Close     |  +100 |
| Medium    |   +50 |
| Far       |    +0 |
| Center    |    +2 |
| Left      |    +0 |
| Right     |    +0 |

The center bonus is only used as a tiebreaker between hazards at the same distance.

A medium-distance hazard can therefore never outrank a close hazard.

The system evaluates all detections in the current frame and selects **one highest-priority hazard**.

---

# Command Mapping

The selected hazard is converted into a two-character command.

The two characters represent independent dimensions:

* **Character 1:** urgency / distance → controls vibration
* **Character 2:** direction → controls audio

| Action        | Score | Command | Haptic Feedback                    | Audio   |
| ------------- | ----: | :-----: | ---------------------------------- | ------- |
| Close Left    | ≥ 100 |   `CL`  | Fast pulse: 70 ms ON / 70 ms OFF   | "left"  |
| Close Center  | ≥ 100 |   `CC`  | Fast pulse: 70 ms ON / 70 ms OFF   | "ahead" |
| Close Right   | ≥ 100 |   `CR`  | Fast pulse: 70 ms ON / 70 ms OFF   | "right" |
| Medium Left   | 50–99 |   `ML`  | Slow pulse: 150 ms ON / 350 ms OFF | "left"  |
| Medium Center | 50–99 |   `MC`  | Slow pulse: 150 ms ON / 350 ms OFF | "ahead" |
| Medium Right  | 50–99 |   `MR`  | Slow pulse: 150 ms ON / 350 ms OFF | "right" |
| Safe          |  < 50 |   `S`   | Motor off                          | Silence |

The Raspberry Pi Pico performs the two lookups independently, allowing the vibration pattern and audio direction to remain synchronized to the same committed hazard state.

---

# Temporal Stabilization

Raw frame-by-frame object detection is inherently noisy.

Bounding boxes can move around distance thresholds, detections can disappear for individual frames, and objects can cross the boundaries between left, center, and right.

Without stabilization, this produced rapid command changes and caused the audio module to repeatedly interrupt itself.

SmartCane therefore uses **asymmetric command hysteresis**.

## Escalation

A more urgent hazard must persist for:

```text
2 consecutive frames
```

At approximately 15 FPS, this corresponds to about:

```text
0.13 seconds
```

This allows the system to respond quickly when a hazard becomes more urgent.

## De-escalation

A less urgent state or lateral direction change must persist for:

```text
8 consecutive frames
```

At approximately 15 FPS:

```text
0.53 seconds
```

This prevents rapid oscillation between states.

The asymmetry reflects the safety priorities of the system:

> The cane should be quick to warn and slow to reassure.

---

## Direction Stabilization

Direction changes at the same urgency level also use the slower eight-frame path.

For example, if an object moves around the boundary between CENTER and RIGHT, the audio does not immediately alternate between:

```text
"ahead"
"right"
"ahead"
"right"
```

Instead, the new direction must remain stable for eight frames before the command changes.

This significantly reduces audio chatter.

---

## Channel Coupling

Both feedback channels originate from the same committed command state.

Therefore:

```text
Committed State
       |
       +------> Vibration
       |
       +------> Audio Direction
```

The vibration and audio channels cannot independently drift into different hazard states.

A direction-only change updates the spoken direction without changing the established vibration urgency.

---

# UDP Communication

The host communicates with the Raspberry Pi Pico using **UDP over WiFi**.

UDP was selected for lightweight, low-latency communication between the host and microcontroller.

The host sends the current two-character hazard command as a UDP datagram.

Example:

```text
CC
```

indicates:

```text
Close + Center
```

while:

```text
MR
```

indicates:

```text
Medium + Right
```

and:

```text
S
```

indicates:

```text
Safe
```

---

## Command Retransmission

Because UDP does not guarantee packet delivery, the host retransmits the current command every:

```text
1 second
```

even when the hazard state has not changed.

This provides a simple liveness mechanism for the microcontroller.

---

# Link Watchdog

The Raspberry Pi Pico maintains a watchdog timer based on the arrival of valid UDP packets.

If no valid command is received for:

```text
3 seconds
```

the Pico automatically returns to the safe state.

This:

* Stops the vibration motor
* Stops active warning behavior
* Protects against wireless dropout
* Handles host crashes
* Handles host sleep
* Handles host battery exhaustion

The three-second timeout provides a margin over the one-second retransmission interval, allowing multiple consecutive packets to be missed before entering the safe state.

This is an important safety feature because UDP itself provides no connection or liveness guarantee.

---

# Image Restoration Study

A major component of the project investigated whether image restoration could improve object detection under motion blur and other camera degradation.

Two detectors were evaluated:

* **YOLOv8n** — lightweight detector used by the working prototype
* **YOLO11s** — stronger detector used to investigate whether restoration effects depended on detector capacity

The restoration and temporal experiments are archived under:

[`denoise_filter/`](denoise_filter/)

The evaluated preprocessing methods are **not part of the live SmartCane pipeline**.

---

# Datasets

## Dataset A — Controlled Synthetic Degradation

Dataset A consisted of:

* 40 clean COCO images
* YOLO-format annotations
* Four synthetic degradation types
* Three severity levels

The degradation types were:

* Gaussian noise
* Motion blur
* Low light
* Glare

Because the original clean images were available, restoration fidelity could be evaluated against a known reference.

For example, the synthetic motion-blur kernels were known exactly, allowing Wiener deconvolution to be supplied with the correct kernel.

---

## Dataset B — Real Cane-Camera Footage

Dataset B consisted of:

* 7 cane-sweep videos
* 84 manually annotated PNG frames
* 344 object instances

The dataset was divided according to perceived blur severity:

| Condition     | Frames | Ground-Truth Instances |
| ------------- | -----: | ---------------------: |
| Clear         |     28 |                    125 |
| Moderate blur |     35 |                    154 |
| Severe blur   |     21 |                     65 |
| **Total**     | **84** |                **344** |

The real dataset is dominated by chairs, which is an important limitation when interpreting aggregate performance.

A variable-frame-rate issue in one video was resolved by matching frames using presentation timestamps rather than assuming fixed frame indices.

---

# Restoration Methods

The following preprocessing approaches were evaluated against an unprocessed raw baseline:

* Gaussian filtering
* Bilateral filtering
* Wiener denoising
* Wiener deconvolution
* CLAHE

For synthetic motion blur, Wiener deconvolution was provided with the **true blur kernel**.

For real cane footage, the blur kernel was unknown.

---

# Controlled Restoration Results

Under controlled synthetic degradation, restoration substantially improved detection.

For YOLOv8n:

| Condition             | Raw mAP@0.5 | Best Restored |
| --------------------- | ----------: | ------------: |
| Clean                 |      0.4255 |        0.4255 |
| Strong Gaussian noise |      0.0860 |        0.1675 |
| Mild motion blur      |      0.1542 |        0.3178 |
| Medium motion blur    |      0.0677 |        0.1998 |
| Severe motion blur    |      0.0464 |        0.0906 |
| Medium low light      |      0.1375 |        0.2166 |
| Severe low light      |      0.0005 |        0.0027 |

For YOLO11s, the same general pattern appeared:

| Condition             | Raw mAP@0.5 | Best Restored |
| --------------------- | ----------: | ------------: |
| Strong Gaussian noise |      0.1685 |        0.2567 |
| Mild motion blur      |      0.3271 |        0.4334 |
| Medium motion blur    |      0.1960 |        0.3128 |
| Severe motion blur    |      0.0774 |        0.1986 |
| Medium low light      |      0.2355 |        0.3491 |
| Severe low light      |      0.0009 |        0.0404 |

These results demonstrate that restoration can be highly effective when the degradation is controlled and known.

However, clean images were not improved by preprocessing, and visual-quality improvements did not consistently predict downstream detection improvements.

---

# Real Cane-Camera Results

The real-world results were substantially different from the synthetic benchmark.

## YOLOv8n

| Method               | Precision | Recall | mAP@0.5 |
| -------------------- | --------: | -----: | ------: |
| Raw                  |    0.7667 | 0.0669 |  0.0504 |
| Gaussian             |    0.7419 | 0.0669 |  0.0338 |
| Bilateral            |    0.6400 | 0.0465 |  0.0279 |
| Wiener denoise       |    0.7037 | 0.0552 |  0.0300 |
| Wiener deconvolution |    0.0000 | 0.0000 |  0.0000 |
| CLAHE                |    0.5686 | 0.0843 |  0.0510 |

CLAHE increased YOLOv8n mAP only slightly:

```text
0.0504 → 0.0510
```

while reducing precision:

```text
0.7667 → 0.5686
```

The 0.0006 mAP difference is too small to constitute convincing evidence of improvement on this limited dataset.

---

## YOLO11s

| Method               | Precision | Recall | mAP@0.5 |
| -------------------- | --------: | -----: | ------: |
| Raw                  |    0.8738 | 0.2616 |  0.2591 |
| Gaussian             |    0.8105 | 0.2238 |  0.1895 |
| Bilateral            |    0.8312 | 0.1860 |  0.1777 |
| Wiener denoise       |    0.8778 | 0.2297 |  0.2227 |
| Wiener deconvolution |    0.0000 | 0.0000 |  0.0000 |
| CLAHE                |    0.7129 | 0.2093 |  0.1830 |

Raw YOLO11s outperformed every tested single-frame preprocessing method.

Recall also decreased substantially with increasing blur:

```text
Clear:          0.3760
Moderate blur:  0.2403
Severe blur:    0.0923
```

This indicates that detected objects were generally correct, but an increasing number of objects were missed as blur became more severe.

---

# Why Restoration Failed on Real Footage

Synthetic motion blur can be represented using a known global degradation model.

Real cane motion is substantially more complex.

The camera may simultaneously experience:

* Rotation
* Translation
* Changing viewpoint
* Objects at different depths
* Partial object visibility
* Spatially varying blur

Consequently, there is no single global blur kernel that accurately describes an entire real cane-camera frame.

Wiener deconvolution therefore performed extremely poorly on real footage:

```text
YOLOv8n:
Precision = 0.0000
Recall    = 0.0000
mAP@0.5  = 0.0000

YOLO11s:
Precision = 0.0000
Recall    = 0.0000
mAP@0.5  = 0.0000
```

### Finding

> Restoration methods that succeed under controlled synthetic degradation do not necessarily generalize to real cane-mounted camera footage.

For this reason, image restoration was removed from the live SmartCane pipeline.

---

# Temporal Image Fusion

The project next investigated whether information from neighboring video frames could recover information lost through motion blur.

Farneback optical flow was used to align neighboring frames with the target frame.

Two fusion strategies were evaluated.

## Fixed Fusion

```text
I_fused =
0.25 I(t-1)
+ 0.50 I(t)
+ 0.25 I(t+1)
```

## Quality-Weighted Fusion

A second method used sharpness-based weighting and alignment-quality checks to determine how much neighboring frames contributed.

---

# Temporal Fusion Results

Pixel-level temporal fusion reduced detection performance.

For YOLO11s:

| Method                      | Precision | Recall | mAP@0.5 |
| --------------------------- | --------: | -----: | ------: |
| Raw                         |    0.8738 | 0.2616 |  0.2591 |
| Temporal – Fixed            |    0.6786 | 0.1657 |  0.2101 |
| Temporal – Quality Weighted |    0.5747 | 0.1453 |  0.1953 |

The effect was particularly severe under heavy blur.

For severe-blur frames:

```text
Raw mAP@0.5:
0.127

Fixed fusion:
0.006

Quality-weighted fusion:
0.000
```

Fixed fusion recovered only 3 detections while losing 25 previously correct detections.

Quality-weighted fusion recovered 2 while losing 28.

The likely cause is inaccurate local optical-flow alignment during large cane movements.

A fused image can appear globally reasonable while still smearing object boundaries enough to remove features needed by YOLO.

### Finding

> Pixel-level temporal fusion is not sufficiently reliable under the large and irregular camera motion produced by normal cane sweeping.

---

# Object-Level Temporal Persistence

Instead of modifying the input image, temporal information was applied **after object detection**.

The motivation was to determine whether an object missed in the current frame could still be detected in nearby frames.

---

## Temporal Oracle Analysis

For YOLOv8n, recall increased from:

```text
Raw target frame: 0.0669
```

to:

```text
Bidirectional ±5 frames: 0.1715
```

| Temporal Window         | Recall |
| ----------------------- | -----: |
| Raw target frame        | 0.0669 |
| Past 1 frame            | 0.0814 |
| Past 3 frames           | 0.1017 |
| Past 5 frames           | 0.1134 |
| Bidirectional ±5 frames | 0.1715 |

For YOLO11s:

| Temporal Window  | Recall |
| ---------------- | -----: |
| Raw target frame | 0.2616 |
| ±1 frame         | 0.3459 |
| ±3 frames        | 0.4390 |
| ±5 frames        | 0.4826 |

A causal, past-only analysis reached:

```text
0.4186 recall
```

at a five-frame history, demonstrating that useful information remains available without using future frames.

---

# ByteTrack and BoT-SORT

Two object-tracking approaches were evaluated:

* ByteTrack
* BoT-SORT

A safety rule was enforced:

> **Current raw detections are never removed.**

All current-frame YOLO detections are passed through unchanged.

If an object briefly disappears, the tracker can temporarily carry its predicted state forward.

---

## YOLOv8n Results

| Method             | Precision | Recall | mAP@0.5 | Rescued / Lost |
| ------------------ | --------: | -----: | ------: | -------------: |
| Raw                |    0.7667 | 0.0669 |  0.0504 |          0 / 0 |
| ByteTrack +1 frame |    0.7742 | 0.0698 |  0.0511 |          1 / 0 |
| BoT-SORT +3 frames |    0.7812 | 0.0727 |  0.0518 |          2 / 0 |

---

## YOLO11s Results

The most useful practical configuration was **ByteTrack with one-frame persistence**.

| Metric              |    Raw | ByteTrack +1 Frame |
| ------------------- | -----: | -----------------: |
| Precision           | 0.8738 |             0.8716 |
| Recall              | 0.2616 |             0.2762 |
| F1                  | 0.4027 |             0.4194 |
| mAP@0.5             | 0.2591 |             0.2698 |
| True Positives      |     90 |                 95 |
| False Positives     |     13 |                 14 |
| Rescued Objects     |      0 |                  5 |
| Lost Raw Detections |      0 |                  0 |

The one-frame persistence configuration recovered five missed objects while preserving all 90 correct raw detections and adding only one stale false positive.

Longer persistence windows introduced additional stale predictions with limited additional recall improvement.

### Finding

> Temporal information is more useful as short-lived object-level state than as pixel-level image fusion.

**Important:** ByteTrack/BoT-SORT persistence has been experimentally evaluated but is **not yet integrated into the live SmartCane pipeline**.

---

# Detector Comparison

Detector capacity had a larger effect on real cane-camera performance than the tested preprocessing methods.

On the same real dataset:

| Detector | Precision | Recall | mAP@0.5 |
| -------- | --------: | -----: | ------: |
| YOLOv8n  |    0.7667 | 0.0669 |  0.0504 |
| YOLO11s  |    0.8738 | 0.2616 |  0.2591 |

The improvement from changing detectors was substantially larger than the gains obtained from preprocessing.

This suggests that future development should prioritize:

* Stronger detectors
* Fine-tuning on cane-camera footage
* Realistic motion-blurred training data
* Model compression
* Edge deployment

rather than relying on universal image preprocessing.

---

# Testing

## Haptic Testing

Two everyday objects were used:

* Bottle
* Cellphone

Testing covered:

* Individual objects on the left
* Individual objects in the center
* Individual objects on the right
* Two objects at matching distances
* Two objects at different distances
* Side-by-side objects
* Different combinations of distance and direction

The system correctly selected the highest-priority hazard when multiple objects were present.

For example:

```text
Bottle:
Close + Center

Cellphone:
Medium + Right

↓

Bottle receives priority

↓

Command: CC

↓

Fast, high-intensity vibration
Audio: "ahead"
```

---

# Audio and Full-System Testing

Audio and physical-assembly testing followed a staged testing ladder so that each stage introduced one new variable.

### 1. DFPlayer and Speaker

The DFPlayer was tested independently using its standalone button mode.

This verified:

* Power delivery
* SD-card readability
* File compatibility
* Speaker continuity

without involving the microcontroller.

### 2. Serial Control

A short script issued UART commands to verify:

* Frame construction
* Checksum calculation
* Baud configuration
* Track playback

### 3. Track Mapping

Each stored audio clip was played sequentially to confirm that the DFPlayer track indices matched the intended direction labels.

### 4. Network Path

Individual command codes were transmitted from the host as isolated UDP datagrams while monitoring the Pico.

This verified that each command generated the correct:

* Vibration pattern
* Audio clip

### 5. Complete System

Finally, the full system was exercised with obstacles at different:

* Distances
* Horizontal positions
* Combinations

The reported command, vibration pattern, and spoken direction were checked against the on-screen detection output.

---

# Challenges

## 1. Unknown Motion-Blur Kernel

Wiener deconvolution depends on knowledge of the degradation operator.

Under synthetic blur, the kernel is known by construction.

Under real cane motion, no single global blur kernel accurately describes the frame.

This caused Wiener deconvolution to collapse to:

```text
Precision = 0.0000
Recall    = 0.0000
mAP@0.5  = 0.0000
```

on the real dataset.

---

## 2. Optical Flow Under Fast Motion

Farneback optical flow worked poorly during large sweeping cane movements.

Misalignment between neighboring frames caused object boundaries to smear during temporal fusion.

This reduced detection performance rather than improving it.

---

## 3. Small and Imbalanced Dataset

The real evaluation dataset contains:

```text
84 frames
7 videos
344 object instances
```

The dataset is also dominated by chairs.

Therefore, aggregate precision, recall, and mAP values may not generalize to the full range of obstacles a cane user could encounter.

Small differences, such as the 0.0006 mAP change observed with CLAHE for YOLOv8n, should therefore not be interpreted as reliable improvements.

---

## 4. Motor Not Stopping on Program Exit

Early testing showed that stopping the host program could leave the vibration motor running because the Pico retained the last received command.

This was addressed with two layers of protection:

1. The host attempts to send the safe command during shutdown.
2. The Pico uses a three-second watchdog to automatically return to the safe state if communication stops.

The watchdog is particularly important because it protects against failures where the host cannot send a final shutdown command.

---

## 5. Close-Range Detection

When an object approaches extremely close to the camera, it can fill or extend beyond the field of view.

This can cause:

* Object truncation
* Extreme aspect-ratio changes
* Loss of recognizable features
* Missed detections

This creates an important edge case where an obstacle can become most dangerous at the same time that its visual representation becomes least recognizable to the detector.

Future versions should investigate fallback proximity heuristics, such as bounding-box area expansion or dedicated proximity sensors.

---

## 6. Audio Clip Truncation

Two separate causes of audio clipping were identified.

The first was rapid state churn, which was addressed through hysteresis.

The second occurred when the safe state explicitly stopped an active audio clip after an obstacle left the frame.

The firmware was revised so that clearing the path allows the current audio clip to finish.

Direction changes can still interrupt an existing clip when the previous direction becomes incorrect.

---

## 7. Network Addressing

The wireless link required the host and Raspberry Pi Pico to share a compatible IPv4 subnet.

Testing through a mobile hotspot produced an address-family mismatch in which the host received an IPv6 transition address while the Pico received a conventional IPv4 address.

The system was therefore tested using a conventional wireless router that assigned both devices addresses on the same IPv4 subnet.

---

# Changes from the Original Plan

The original project proposed a dual-device wearable system consisting of:

* A smart cane for lower obstacles
* Smart glasses for overhead hazards
* Multiple camera feeds
* Centralized processing
* Frequency-domain image restoration

During development, several practical limitations were identified.

### Image Restoration

Restoration improved detection under controlled synthetic degradation but did not generalize to real cane-camera footage.

### Dual Camera Feeds

Multiple wireless camera feeds introduced:

* Additional latency
* Dropped frames
* Connection instability

### Continuous Audio

Announcing every detected object generated excessive feedback and interfered with communication of urgent hazards.

### Final Architecture

The project was therefore simplified to a single-camera SmartCane.

The smart-glasses component was moved to future work, and image restoration was removed from the live pipeline.

The final live system prioritizes:

```text
Camera
  ↓
YOLOv8n
  ↓
Hazard Scoring
  ↓
Priority Selection
  ↓
Hysteresis
  ↓
UDP / WiFi
  ↓
Raspberry Pi Pico
  ↓
Haptic + Directional Audio
```

---

# Experiment Archive

The complete image-restoration, temporal-fusion, oracle, and tracking experiments are stored under:

```text
denoise_filter/
```

The experiment directory contains its own source code, datasets, annotations, stored results, analysis scripts, and requirements.

Run experimental commands from that directory:

```bash
cd denoise_filter
```

The live SmartCane application remains at the repository root.

The restoration and tracking methods are research/evaluation components and are not currently part of the deployed live hardware pipeline.

---

# Future Work

Potential improvements include:

### Detector Improvements

* Fine-tuning YOLOv8n or a stronger detector on real cane-camera footage
* Investigating YOLO11s or other stronger lightweight detectors
* Model compression for edge deployment
* Training with realistic motion-blurred cane-camera data

### Temporal Processing

* Integrating one-frame ByteTrack persistence into the live pipeline
* Investigating safer temporal confidence handling
* Developing causal tracking methods with low latency

### Hardware Improvements

* Higher-frame-rate camera
* Global-shutter camera
* Improved camera stabilization
* More rigid camera mounting
* Depth camera for true distance estimation
* Dedicated proximity sensing for extreme close-range obstacles

### Edge Deployment

Moving inference from the host computer to an edge-AI device such as:

* Raspberry Pi 5
* Other AI-capable embedded hardware

This would reduce the system's dependence on an external laptop.

### Dataset Expansion

Future evaluation should include:

* More videos
* More object classes
* Outdoor environments
* Different lighting conditions
* Different camera motions
* More diverse obstacle types
* More balanced class distributions

### User Evaluation

Formal testing with visually impaired participants is required to determine:

* Whether vibration patterns are intuitive
* Whether audio directions are understandable
* Whether distance thresholds are appropriate
* Whether the alert timing is useful during real navigation
* Whether the system introduces cognitive overload

---

# Key Conclusions

The project produced four primary conclusions:

1. **Image restoration substantially improves detection under controlled synthetic degradation.**

2. **These improvements do not reliably transfer to real cane-mounted camera footage**, where motion blur is spatially varying and the degradation kernel is unknown.

3. **Pixel-level temporal fusion can reduce detection performance** when optical-flow alignment fails during large camera movements.

4. **Short-term object-level persistence is a more promising approach to temporal robustness**, recovering temporary missed detections without removing correct raw detections.

A fifth important finding emerged from the detector comparison:

> **Detector capacity had a substantially larger impact on real cane-camera performance than universal image preprocessing.**

On the same real footage, YOLOv8n achieved:

```text
mAP@0.5 = 0.0504
Recall   = 0.0669
```

while YOLO11s achieved:

```text
mAP@0.5 = 0.2591
Recall   = 0.2616
```

Overall, the experimental evidence suggests that future SmartCane development should prioritize:

```text
Stronger detector
       +
Realistic training data
       +
Lightweight post-detection temporal reasoning
       +
Improved camera hardware
```

rather than relying on universal image preprocessing.

---

# Repository Structure

```text
SmartCane/
│
├── README.md
│
├── main.py
├── decision.py
├── command.py
├── hysteresis.py
├── important_objects.py
│   Live host-side detection, hazard scoring,
│   command generation, and temporal stabilization
│
├── main_pico.py
│   Raspberry Pi Pico MicroPython firmware
│
├── audio_files/
│   Audio clips and SD-card preparation utilities
│
└── denoise_filter/
    │
    ├── README.md
    ├── requirements.txt
    │
    ├── restoration/
    ├── temporal/
    ├── tracking/
    ├── analysis/
    │
    ├── data/
    ├── annotation/
    ├── smart_cane_frames/
    └── results/
```

---

# Citation

If referencing this project, use:

```text
K. F. Qaisar, T. Tran, and R. Balroop,
"SmartCane: A Vision-Guided Navigation Aid for Visually Impaired Users,"
2026.
```

---

## Project Status

**Prototype status:** Functional research prototype

The current system demonstrates:

* Real-time YOLO-based obstacle detection
* Hazard prioritization
* Haptic urgency feedback
* Directional audio feedback
* UDP communication
* Microcontroller watchdog protection
* Temporal command stabilization

The restoration and temporal-tracking experiments provide additional research findings that guide future development but are not currently integrated into the live perception pipeline.

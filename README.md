Absolutely. Since this is a **technical project report**, your README should be much shorter than the paper and focus on **what the project is, how to run it, hardware/software requirements, project structure, and key findings**.

Here’s a polished README you can put directly in your GitHub repository:

# SmartCane: A Vision-Guided Navigation Aid for Visually Impaired Users

**Khawaja Faiza Qaisar – 217948233**
**Tran Tran – 220168829**
**Richard Balroop – 216906349**

## Overview

SmartCane is an assistive navigation prototype designed to help visually impaired users detect obstacles before physical contact occurs.

Traditional white canes primarily detect obstacles at ground level through physical contact. SmartCane extends this capability by combining **real-time computer vision, object detection, audio feedback, and haptic vibration alerts**.

The system uses a webcam mounted on a cane to capture the user's surroundings. A host computer runs a lightweight **YOLO object detector** and determines:

* What obstacle has been detected
* Whether the obstacle is on the left, center, or right
* Whether the obstacle is far, medium, or close
* Which detected obstacle represents the most urgent hazard

The resulting hazard state is transmitted over USB serial to a **Raspberry Pi Pico**, which controls the vibration motor and audio feedback.

---

## System Architecture

```text
             Camera
                |
                v
       +------------------+
       |   Host Computer  |
       |                  |
       |   YOLO Detector  |
       |        |         |
       |        v         |
       | Hazard Scoring   |
       |        |         |
       |        v         |
       | Serial Command   |
       +--------+---------+
                |
              USB
                |
                v
       +------------------+
       | Raspberry Pi Pico|
       |                  |
       |   MicroPython    |
       +--------+---------+
                |
          +-----+-----+
          |           |
          v           v
     Vibration      Audio
       Motor        Module
```

---

## Key Features

* Real-time object detection using YOLO
* Detection of multiple obstacles in a single frame
* Left / center / right hazard classification
* Distance approximation using bounding-box height
* Priority-based hazard selection
* Three-level haptic feedback
* USB serial communication between host computer and Raspberry Pi Pico
* Heartbeat communication for improved safety
* Automatic motor shutdown when the host program exits
* Evaluation of image restoration techniques for motion-blurred cane footage
* Evaluation of temporal detection methods using object tracking

---

## Hardware

### Prototype Hardware

* Raspberry Pi Pico
* USB connection
* Camera / webcam
* Vibration motor
* Audio module
* Cane mounting hardware
* Host computer/laptop for real-time object detection

The Raspberry Pi Pico is responsible for the hardware control layer, while computationally intensive object detection is performed on the host computer.

---

## Software Requirements

### Host Computer

The project requires Python with the appropriate computer vision and machine learning dependencies.

Major components include:

* Python 3
* YOLO / Ultralytics
* OpenCV
* NumPy
* PyTorch
* Serial communication (`pyserial`)

Additional packages may be required for the restoration and tracking experiments.

### Raspberry Pi Pico

The Pico runs:

* MicroPython
* Serial command handling
* GPIO motor control
* Audio control

The MicroPython program can be flashed to the Pico using **Thonny IDE**.

---

## Hazard Detection

Each YOLO detection is converted into a hazard score based on its apparent distance and horizontal position.

### Direction

The image is divided into three columns:

```text
+----------------+----------------+----------------+
|                |                |                |
|      LEFT      |     CENTER     |     RIGHT      |
|                |                |                |
+----------------+----------------+----------------+
```

The horizontal center of a bounding box determines the direction.

### Distance Approximation

Distance is estimated using bounding-box height:

| Distance | Bounding-box height |
| -------- | ------------------: |
| Close    |            > 250 px |
| Medium   |          120–250 px |
| Far      |            ≤ 120 px |

Far objects are ignored by the hazard-scoring system.

### Hazard Score

```text
Hazard Score = Distance Score + Direction Bonus
```

| Condition    | Score |
| ------------ | ----: |
| Close        |  +100 |
| Medium       |   +50 |
| Far          |    +0 |
| Center       |    +2 |
| Left / Right |    +0 |

The center bonus acts only as a tiebreaker between hazards at the same distance.

---

## Haptic Commands

The highest-scoring hazard is converted into a single-character command.

| State | Score | Command | Haptic Feedback                 |
| ----- | ----: | ------- | ------------------------------- |
| STOP  | ≥ 100 | `C`     | Fast, high-intensity pulses     |
| WARN  | 50–99 | `M`     | Slower, medium-intensity pulses |
| SAFE  |  < 50 | `S`     | Motor off                       |

Only the current highest-priority hazard controls the haptic output.

For example, if a bottle is close and centered while a cellphone is farther away on the right, the bottle receives the higher hazard score and determines the alert.

---

## Serial Communication

The host computer communicates with the Raspberry Pi Pico using USB serial.

Three commands are used:

```text
C → STOP / close hazard
M → WARN / medium-distance hazard
S → SAFE / no relevant hazard
```

To improve reliability, the host uses two communication mechanisms:

### State-change transmission

A command is sent when the hazard state changes.

### Heartbeat

The current state is retransmitted every **2 seconds**, even if the state has not changed.

This prevents the Pico from remaining indefinitely in an incorrect state if a serial byte is lost.

### Safe shutdown

The host application uses a `try/finally` shutdown mechanism to send:

```text
S
```

before closing the camera and serial connection.

This ensures that the vibration motor is stopped when the program is interrupted or exits unexpectedly.

---

# Image Restoration Study

A major part of the project investigated whether image restoration could improve object detection under motion blur and other camera degradation.

Two object detectors were evaluated:

* **YOLOv8n** – lightweight detector used by the prototype
* **YOLO11s** – stronger detector used to investigate whether restoration effects depended on detector capacity

## Datasets

### Dataset A – Synthetic Degradation

Dataset A consisted of **40 clean COCO images** with YOLO-format annotations.

Four degradation types were artificially applied at multiple severity levels:

* Gaussian noise
* Motion blur
* Low light
* Glare

Because the original clean images were available, restoration quality could be evaluated against a known reference.

### Dataset B – Real Cane-Camera Footage

Dataset B consisted of:

* 7 cane-sweep videos
* 84 manually annotated frames
* 344 object instances

The frames were categorized as:

| Condition     | Frames | Ground-Truth Instances |
| ------------- | -----: | ---------------------: |
| Clear         |     28 |                    125 |
| Moderate blur |     35 |                    154 |
| Severe blur   |     21 |                     65 |
| **Total**     | **84** |                **344** |

The real dataset is chair-dominated, which should be considered when interpreting aggregate performance.

---

## Restoration Methods

The following preprocessing approaches were evaluated against an unprocessed raw baseline:

* Gaussian filtering
* Bilateral filtering
* Wiener denoising
* Wiener deconvolution
* CLAHE

For synthetic motion blur, Wiener deconvolution was provided with the known blur kernel.

---

## Main Restoration Findings

Restoration produced substantial improvements on **controlled synthetic degradation**.

For example, YOLOv8n mAP@0.5 increased:

```text
Mild motion blur:
0.1542 → 0.3178

Medium motion blur:
0.0677 → 0.1998
```

However, these improvements did **not transfer to real cane-camera footage**.

On real footage, raw YOLO11s achieved:

```text
Precision: 0.8738
Recall:    0.2616
mAP@0.5:   0.2591
```

and outperformed every tested single-frame restoration method.

Wiener deconvolution completely failed on the real footage:

```text
Precision: 0.0000
Recall:    0.0000
mAP@0.5:   0.0000
```

The primary reason is that synthetic motion blur uses a known global degradation model, while real cane motion produces spatially varying blur caused by camera rotation, translation, and objects at different depths.

### Conclusion

> Restoration methods that work under controlled degradation do not necessarily generalize to real cane-mounted camera footage.

For this reason, image restoration was removed from the live SmartCane pipeline.

---

# Temporal Processing

Temporal information was investigated as another method of improving detection under motion blur.

Two approaches were tested:

1. Pixel-level temporal image fusion
2. Object-level temporal persistence

## Pixel-Level Fusion

Neighboring frames were aligned using Farneback optical flow and fused with the target frame.

The fixed fusion strategy was:

```text
0.25 I(t-1) + 0.50 I(t) + 0.25 I(t+1)
```

However, temporal pixel fusion reduced detection performance.

For YOLO11s:

| Method                      | Precision | Recall | mAP@0.5 |
| --------------------------- | --------: | -----: | ------: |
| Raw                         |    0.8738 | 0.2616 |  0.2591 |
| Temporal – Fixed            |    0.6786 | 0.1657 |  0.2101 |
| Temporal – Quality Weighted |    0.5747 | 0.1453 |  0.1953 |

The likely cause is inaccurate local alignment during large cane movements, which can smear object boundaries even when global image-quality metrics appear reasonable.

---

## Object-Level Persistence

Instead of modifying the input image, temporal information was applied after object detection.

ByteTrack and BoT-SORT were evaluated to determine whether an object detected in nearby frames could temporarily persist when the detector missed it.

A key safety rule was enforced:

> Current raw detections are never removed.

For YOLO11s, one-frame ByteTrack persistence produced:

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

This suggests that short-term object persistence is a more promising approach than pixel-level temporal fusion.

---

# Detector Comparison

One of the strongest findings was the difference between detector capacity.

On the same real cane-camera footage:

| Detector | Precision | Recall | mAP@0.5 |
| -------- | --------: | -----: | ------: |
| YOLOv8n  |    0.7667 | 0.0669 |  0.0504 |
| YOLO11s  |    0.8738 | 0.2616 |  0.2591 |

The improvement from changing the detector was substantially larger than the gains obtained through preprocessing.

This suggests that future development should prioritize:

* stronger detectors
* model fine-tuning
* realistic cane-camera training data
* model compression for edge deployment

rather than universal image preprocessing.

---

# Testing

The working haptic system was tested using a bottle and cellphone.

Tests included:

* Individual objects on the left
* Individual objects in the center
* Individual objects on the right
* Two objects at matching distances
* Two objects at different distances
* Side-by-side objects
* Different combinations of distance and direction

The system correctly prioritized the more urgent object when multiple objects were present.

For example:

```text
Bottle:
Close + Center

Cellphone:
Medium + Right

→ Bottle receives priority
→ STOP command (`C`)
→ Fast, high-intensity vibration
```

Audio-module testing, denoising evaluation, and complete physical-assembly testing are still ongoing.

---

# Challenges

Several challenges significantly influenced the final system design.

### 1. Unknown Motion-Blur Kernel

Wiener deconvolution requires knowledge of the degradation kernel. Real cane movement does not produce a single fixed global blur kernel.

### 2. Optical Flow Under Fast Motion

Large sweeping cane movements caused inaccurate optical-flow alignment and introduced image smearing during temporal fusion.

### 3. Limited Real-World Dataset

The real evaluation dataset contains only 84 frames from seven videos and is dominated by chairs.

### 4. Motor Shutdown

Early testing revealed that the motor could continue vibrating after the host program stopped. A `try/finally` shutdown mechanism was added to guarantee that the Pico receives the `S` command.

### 5. Close-Range Detection

When an object approaches extremely close to the camera, it may fill or extend beyond the camera's field of view. This can cause truncation and loss of recognizable features, resulting in missed detections at precisely the point where an obstacle is most urgent.

---

# Changes from the Original Plan

The original design proposed a dual-device system consisting of:

* A smart cane for lower obstacles
* Smart glasses for overhead obstacles
* Multiple camera feeds
* Centralized processing
* Frequency-domain image restoration

During development, several issues were identified:

* Restoration did not generalize to real cane footage
* Dual wireless camera feeds introduced latency and connection instability
* Continuous audio announcements created excessive feedback
* Additional preprocessing increased computational cost

The final prototype therefore uses a **single camera mounted on the cane**, with the smart-glasses component reserved for future development.

The live pipeline prioritizes:

```text
Camera
  ↓
YOLO Detection
  ↓
Hazard Scoring
  ↓
Priority Selection
  ↓
USB Serial
  ↓
Raspberry Pi Pico
  ↓
Haptic / Audio Alert
```

---

# Future Work

Potential improvements include:

* Fine-tuning a stronger detector on real cane-camera footage
* Integrating ByteTrack persistence into the live pipeline
* Using realistic motion-blurred training data
* Testing higher-frame-rate cameras
* Using a global-shutter camera
* Improving camera stabilization and mounting
* Replacing bounding-box size with true depth sensing
* Moving inference onto an edge-AI device such as a Raspberry Pi 5
* Expanding the real-world dataset
* Testing outdoors and in more diverse environments
* Evaluating additional obstacle classes
* Conducting formal user testing with visually impaired participants
* Optimizing vibration patterns and audio feedback
* Investigating fallback proximity heuristics for extreme close-range obstacles

---

# Key Conclusions

The project produced four main conclusions:

1. **Image restoration can substantially improve detection under controlled synthetic degradation.**
2. **These improvements do not reliably transfer to real cane-mounted footage.**
3. **Pixel-level temporal fusion can hurt detection when camera motion causes inaccurate alignment.**
4. **Short-term object-level persistence provides a safer and more practical approach to recovering temporary missed detections.**

Overall, the experiments suggest that SmartCane should prioritize **detector capacity, realistic training data, and lightweight post-detection temporal reasoning** rather than relying on universal image preprocessing.

---

## Repository Structure

A recommended repository structure is:

```text
SmartCane/
│
├── README.md
│
├── host/
│   ├── detection.py
│   ├── hazard_scoring.py
│   └── serial_control.py
│
├── pico/
│   └── main.py
│
├── restoration/
│   ├── synthetic_experiments.py
│   ├── real_footage_experiments.py
│   └── temporal_fusion.py
│
├── tracking/
│   └── temporal_persistence.py
│
├── datasets/
│   └── README.md
│
├── results/
│   ├── tables/
│   └── figures/
│
└── report/
    └── SmartCane_Report.pdf
```

Adjust the filenames and folders to match the actual repository contents.

---

## Citation

If referencing this project, use:

```text
K. F. Qaisar, T. Tran, and R. Balroop,
"SmartCane: A Vision-Guided Navigation Aid for Visually Impaired Users,"
2026.
```

---

## Status

**Prototype Status:** Working prototype

**Object Detection:** Implemented
**Hazard Prioritization:** Implemented
**Haptic Feedback:** Implemented
**Serial Communication:** Implemented
**Safety Shutdown:** Implemented
**Image Restoration Study:** Completed
**Temporal Detection Study:** Completed
**Live Object Persistence:** Future integration
**Audio Testing:** Ongoing
**Full Physical Assembly Testing:** Ongoing

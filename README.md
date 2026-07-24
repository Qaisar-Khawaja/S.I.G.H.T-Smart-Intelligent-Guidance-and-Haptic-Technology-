# Smart Cane for the Visually Impaired

A Computer Vision based Smart Cane that detects nearby obstacles using **YOLOv8** and provides real-time feedback through a **Raspberry Pi Pico 2 W**.

The goal of this project is to assist visually impaired users by identifying nearby obstacles, determining their direction (left, center, or right), estimating how close they are, and providing appropriate feedback.

---

# Features

- Real-time object detection using YOLOv8
- Detects important obstacles such as:
  - Person
  - Chair
  - Bicycle
  - Car
  - Dog
- Determines obstacle direction:
  - Left
  - Center
  - Right
- Estimates obstacle distance using bounding box size
- Makes navigation decisions
- Sends commands to Raspberry Pi Pico 2 W
- Controls hardware feedback (currently onboard LED)

---

# Current System Architecture

```
                Laptop Camera
                     │
                     ▼
             YOLOv8 Object Detection
                     │
                     ▼
             Direction Detection
                     │
                     ▼
             Distance Estimation
                     │
                     ▼
              Decision Logic
                     │
                     ▼
             Command Generation
                     │
                 USB Serial
                     │
                     ▼
          Raspberry Pi Pico 2 W
                     │
                     ▼
             LED / Future Motors
```

# Software

- Python 3.11
- MicroPython
- Thonny IDE
- OpenCV
- Ultralytics YOLOv8
- PyTorch
- NumPy
- PySerial
---

# Repository Structure

```
SmartCane/
│
├── main.py
├── decision.py
├── command.py
├── pico_connection.py
├── README.md
└── requirements.txt
```

### File Description

### `main.py`

Main application.

Responsibilities:

- Open camera
- Run YOLO detection
- Determine obstacle direction
- Estimate obstacle distance
- Call decision logic
- Generate hardware commands
- Send commands to Pico

---

### `decision.py`

Contains the smart cane navigation logic.

Input:

- Object
- Direction
- Distance

Output:

- WARN_LEFT
- WARN_RIGHT
- STOP
- SAFE

---

### `command.py`

Converts actions into hardware commands.

| Action | Command |
|---------|----------|
| WARN_LEFT | L |
| WARN_RIGHT | R |
| STOP | C |
| SAFE | S |

---

### `pico_connection.py`

Handles USB serial communication between the laptop and Raspberry Pi Pico 2 W.

---

# Installation

## Step 1 - Clone Repository

```bash
git clone https://github.com/USERNAME/SmartCane.git
cd SmartCane
```

---

## Step 2 - Create Virtual Environment

Python 3.11 is recommended.

```bash
python3.11 -m venv venv
```

Activate environment

macOS/Linux

```bash
source venv/bin/activate
```

Windows

```bash
venv\Scripts\activate
```

---

## Step 3 - Install Dependencies

```bash
pip install ultralytics
pip install opencv-python
pip install pyserial
pip install torch torchvision torchaudio
```

---

# Installation Issues Encountered

## Issue 1 - Python 3.13

Initially the project was created using Python 3.13.

Installing PyTorch produced:

```
ERROR: No matching distribution found for torch
```

### Solution

Create a Python 3.11 virtual environment.

---

## Issue 2 - NumPy Version

PyTorch produced the warning:

```
A module compiled using NumPy 1.x cannot be run in NumPy 2.x
```

### Solution

Downgrade NumPy.

```bash
pip install "numpy<2"
```

Installed version:

```
NumPy 1.26.4
```

---

# Verify Installation

Verify PyTorch

```bash
python -c "import torch; print(torch.__version__)"
```

Verify YOLO

```bash
python -c "from ultralytics import YOLO; print('YOLO works')"
```

Expected output

```
YOLO works
```

---

# Object Detection

YOLOv8 detects objects from the webcam.

Example workflow

```
Camera
    ↓
YOLO
    ↓
Person detected
```

Only important objects are considered.

Current object list

- Person
- Chair
- Bicycle
- Car
- Dog

---

# Direction Detection

Each frame is divided into three equal sections.

```
LEFT | CENTER | RIGHT
```

The object's bounding box center determines its direction.

```
center_x = (x1 + x2) / 2
```

---

# Distance Estimation

The project estimates distance using the height of the detected bounding box.

```
Height > 250
    CLOSE

Height > 120
    MEDIUM

Otherwise
    FAR
```

This is an approximation.

Larger bounding boxes usually indicate objects that are closer to the camera.

---

# Decision Logic

Examples

```
Person
Center
Close

↓

STOP
```

```
Chair
Left
Close

↓

WARN_LEFT
```

```
Dog
Right
Close

↓

WARN_RIGHT
```

---

# Command Generation

The generated action is converted into a simple command.

```
WARN_LEFT
↓

L
```

```
WARN_RIGHT
↓

R
```

```
STOP
↓

C
```

```
SAFE
↓

S
```

---

# Raspberry Pi Pico 2 W

MicroPython was installed on the Pico.

Thonny IDE was used to upload programs.

The Pico receives commands through USB serial.

Example

```
Received: C
```

The onboard LED responds to received commands.

---

# Communication Pipeline

Current communication

```
Laptop
    │
USB Serial
    │
Pico 2 W
```

Future communication

```
Laptop
    │
WiFi
    │
Pico 2 W
```

---

# Current Project Status

Completed

- Computer Vision pipeline
- YOLOv8 object detection
- Direction detection
- Distance estimation
- Decision logic
- Command generation
- Raspberry Pi Pico communication
- LED feedback

---

# Future Improvements

- Wireless communication using WiFi
- External LEDs
- Vibration motors
- Audio feedback
- Speaker with voice alerts
- Better distance estimation
- Walking path detection
- Object prioritization
- Outdoor testing

---

# Technologies Used

- Python
- MicroPython
- OpenCV
- Ultralytics YOLOv8
- PyTorch
- Raspberry Pi Pico 2 W
- Computer Vision
- Embedded Systems
- Serial Communication

---

# Project Workflow

```
Camera
   │
   ▼
YOLOv8
   │
   ▼
Object Detection
   │
   ▼
Direction Detection
   │
   ▼
Distance Estimation
   │
   ▼
Decision Logic
   │
   ▼
Command Generation
   │
   ▼
USB Serial
   │
   ▼
Raspberry Pi Pico 2 W
   │
   ▼
LED Feedback
```

---

# Setup Instructions

Follow the steps below to set up the Smart Cane project from scratch.

---

# Step 1 - Create Project Folder

Open Terminal.

Create a new folder for the project.

```bash
mkdir SmartCane
cd SmartCane
```

---

# Step 2 - Create a Python Virtual Environment

> **Recommended Python Version: 3.11**

Do **NOT** use Python 3.13 because PyTorch may not install correctly.

Create the virtual environment:

```bash
python3.11 -m venv venv
```

Activate the virtual environment.

macOS/Linux

```bash
source venv/bin/activate
```

Windows

```bash
venv\Scripts\activate
```

Your terminal should now show:

```
(venv)
```

---

# Step 3 - Upgrade pip

```bash
pip install --upgrade pip
```

---

# Step 4 - Install Required Libraries

Install OpenCV

```bash
pip install opencv-python
```

Install YOLO

```bash
pip install ultralytics
```

Install serial communication library

```bash
pip install pyserial
```

Install PyTorch

```bash
pip install torch torchvision torchaudio
```

---

# Step 5 - NumPy Compatibility Fix

If you receive the warning

```
A module compiled using NumPy 1.x cannot be run in NumPy 2.x
```

install NumPy 1.x

```bash
pip install "numpy<2"
```

---

# Step 6 - Verify Installation

Check PyTorch

```bash
python -c "import torch; print(torch.__version__)"
```

Check YOLO

```bash
python -c "from ultralytics import YOLO; print('YOLO works')"
```

Expected output

```
YOLO works
```

---

# Step 7 - Create Project Files

Create the following files inside the project folder.

```
SmartCane/
│
├── main.py
├── decision.py
├── command.py
├── pico_connection.py
└── README.md
```

---

# Step 8 - Add the Python Code

Copy the project code into

- `main.py`
- `decision.py`
- `command.py`
- `pico_connection.py`

Save all files.

---

# Step 9 - Download the YOLO Model

The first time the following line is executed

```python
model = YOLO("yolov8n.pt")
```

YOLO automatically downloads

```
yolov8n.pt
```

No manual download is required.

---

# Step 10 - Test Object Detection

Run

```bash
python main.py
```

The laptop webcam should open.

Verify that objects such as

- Person
- Chair
- Dog
- Bicycle

are detected.

Close the window by pressing

```
q
```

---

# Step 11 - Install MicroPython on Raspberry Pi Pico 2 W

Download the latest MicroPython firmware for the Raspberry Pi Pico 2 W from the official MicroPython website.

Disconnect the Pico from your computer.

Press and hold the **BOOTSEL** button on the Pico.

While holding the button:

- Connect the Pico to the computer using USB.
- Release the BOOTSEL button.

A new USB drive named

```
RPI-RP2
```

should appear.

Copy the downloaded `.uf2` firmware file onto the `RPI-RP2` drive.

The Pico will automatically reboot with MicroPython installed.

---

# Step 12 - Install Thonny IDE

Download and install Thonny.

Open Thonny.

Go to

```
Tools
→ Options
→ Interpreter
```

Select

```
MicroPython (Raspberry Pi Pico)
```

Choose the Pico USB port.

Click

```
OK
```

You should now see the MicroPython shell

```
>>>
```

---

# Step 13 - Test the Pico

Create a new Python file in Thonny.

Example

```python
from machine import Pin
import time

led = Pin("LED", Pin.OUT)

while True:
    led.toggle()
    time.sleep(1)
```

Save the file as

```
main.py
```

Choose

```
Raspberry Pi Pico
```

as the save location.

The onboard LED should begin blinking.

---

# Step 14 - Find the Pico USB Port

Disconnect and reconnect the Pico.

Open Terminal.

Run

```bash
ls /dev/cu.*
```

Example output

```
/dev/cu.usbmodem144201
```

Copy this port name.

---

# Step 15 - Configure Serial Communication

Open

```
pico_connection.py
```

Replace the serial port with your Pico port.

Example

```python
serial.Serial(
    "/dev/cu.usbmodem144201",
    115200,
    timeout=1
)
```

Save the file.

---

# Step 16 - Upload Pico Receiver Code

Open Thonny.

Replace the Pico code with the serial receiver program.

Save it again as

```
main.py
```

on the Pico.

Restart the Pico.

The Pico should print

```
Smart Cane Pico Ready
```

---

# Step 17 - Close Thonny

Before running the computer vision program,

**close Thonny completely.**

Only one application can access the Pico USB port at a time.

If Thonny remains open, Python will not be able to communicate with the Pico.

---

# Step 18 - Run the Smart Cane

Activate the virtual environment

```bash
source venv/bin/activate
```

Run

```bash
python main.py
```

The webcam should open.

When a person or obstacle is detected,

the laptop should print something similar to

```
Object: person
Direction: CENTER
Distance: CLOSE
Action: STOP
Command: C
```

The Pico should receive the command and respond by turning on the onboard LED.

---

# Step 19 - Exit the Program

Press

```
q
```

to close the webcam.

Disconnect the Pico when finished.


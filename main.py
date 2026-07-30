import time
import cv2
from ultralytics import YOLO

from command import generate_command
from decision import cane_decision
from pico_connection import send_command

# Load YOLO
model = YOLO("yolov8n.pt")

# Open camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

important_objects = [
    "person",
    "chair",
    "bicycle",
    "car",
    "dog"
]

previous_command = None
last_time_sent = 0
send_delay = 1

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape

        left_limit = width / 3
        right_limit = 2 * width / 3

        results = model(frame)

        for result in results:
            boxes = result.boxes

            for box in boxes:
                class_id = int(box.cls[0])
                label = model.names[class_id]

                if label not in important_objects:
                    continue

                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                center_x = (x1 + x2) / 2

                # Direction
                if center_x < left_limit:
                    direction = "LEFT"
                elif center_x > right_limit:
                    direction = "RIGHT"
                else:
                    direction = "CENTER"

                # Distance estimation
                box_height = y2 - y1

                if box_height > 250:
                    distance = "CLOSE"
                elif box_height > 120:
                    distance = "MEDIUM"
                else:
                    distance = "FAR"

                # Decision
                action = cane_decision(
                    label,
                    direction,
                    distance
                )

                # Hardware command
                command = generate_command(action)

                current_time = time.time()

                # Avoid sending same command repeatedly
                if command != previous_command:
                    print(
                        f"""
Object: {label}
Direction: {direction}
Distance: {distance}
Action: {action}
Command: {command}
"""
                    )

                    send_command(command)

                    previous_command = command
                    last_time_sent = current_time

        annotated = results[0].plot()

        cv2.imshow(
            "Smart Cane Vision",
            annotated
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # THIS ALWAYS RUNS, EVEN IF YOU PRESS CTRL+C OR THE SCRIPT CRASHES
    print("\nStopping vision system & turning off motor...")
    try:
        send_command("S")  # Turn off motor on Pico
    except Exception as e:
        print("Failed to send stop command:", e)

    cap.release()
    cv2.destroyAllWindows()
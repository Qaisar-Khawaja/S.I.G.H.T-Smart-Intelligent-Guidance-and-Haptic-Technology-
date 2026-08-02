import time
import cv2
from ultralytics import YOLO

from decision import hazard_score, cane_decision
from command import generate_command
from pico_connection import send_command

# Load YOLO
model = YOLO("yolov8n.pt")

# Open camera
# find_camera.py once if you're not sure which index is which.
CAMERA_INDEX = 0
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

important_objects = [
    "bottle",
    "cell phone",
]

previous_command = None
last_time_sent = 0
heartbeat_interval = 2.0  # resend the current command this often even if it
                          # hasn't changed, so a dropped serial byte can't
                          # leave the motor stuck on (or stuck off)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape

        left_limit = width / 3
        right_limit = 2 * width / 3

        results = model(frame)

        # Score every dangerous object in this frame and keep only the
        # single most urgent one. This is what makes "two hazards at once"
        # resolve to the closer/more-central one, instead of whichever box
        # happened to be processed last.
        top_score = 0
        top_info = None  # (label, direction, distance)

        # Collect draw info for every box so we can label pixel height
        # and distance tier on screen -- this is what you use to pick
        # real CLOSE/MEDIUM/FAR thresholds instead of guessing them.
        overlay_boxes = []

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

                score = hazard_score(label, direction, distance)

                if score > top_score:
                    top_score = score
                    top_info = (label, direction, distance)

                overlay_boxes.append(
                    (int(x1), int(y1), int(y2), label, direction, distance, int(box_height))
                )

        # Decide based on the single most urgent hazard this frame
        if top_info:
            label, direction, distance = top_info
            action = cane_decision(label, direction, distance)
        else:
            label, direction, distance = None, None, None
            action = "SAFE"

        # Hardware command
        command = generate_command(action)

        current_time = time.time()

        should_send = (
            command != previous_command
            or (current_time - last_time_sent) >= heartbeat_interval
        )

        if should_send:
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

        # Draw "h=<pixel height> <DISTANCE>" under each box so you can
        # watch the number live and see exactly where it crosses your
        # CLOSE/MEDIUM/FAR cutoffs as you move an object toward the camera.
        for (x1, y1, y2, ov_label, ov_direction, ov_distance, box_height) in overlay_boxes:
            text = f"h={box_height} {ov_distance}"
            text_y = min(y2 + 25, height - 10)  # keep text on-screen even for tall boxes
            cv2.putText(
                annotated,
                text,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

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


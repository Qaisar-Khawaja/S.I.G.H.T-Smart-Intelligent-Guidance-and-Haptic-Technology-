import cv2
from pathlib import Path
from ultralytics import YOLO

from command import generate_command
from decision import cane_decision, hazard_score
from hysteresis import CommandStabilizer
from pico_connection import send_command
import important_objects


PROJECT_DIR = Path(__file__).resolve().parent
model = YOLO(str(PROJECT_DIR / "yolov8n.pt"))

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open camera")
    exit()


DEBUG = False               # one console line per detection

# Owns hysteresis, de-duplication, and the heartbeat resend.
# Tuning lives in hysteresis.py, not here.
stabilizer = CommandStabilizer()


try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        left_limit = width / 3
        right_limit = 2 * width / 3



# 
# Changed Line
        results = model(frame, verbose=False)
# 




        # ── Score every hazard in this frame, keep only the worst ──
        # One command per frame, not one per detection. Sending per
        # detection means a distant chair's SAFE cancels a close
        # person's STOP milliseconds after it was sent.

        top_score = 0
        top_info = None          # (label, direction, distance)
        overlay_boxes = []             # on-screen h=<px> labels

        for result in results:
            for box in result.boxes:            # type: ignore
                label = model.names[int(box.cls[0])]

                if label not in important_objects.important_objects:
                    continue

                x1, y1, x2, y2 = box.xyxy[0]
                center_x = (x1 + x2) / 2
               
                
                
                
                
                # 
                # Changed Line
                box_height = float(y2 - y1)
                # 



                if center_x < left_limit:
                    direction = "LEFT"
                elif center_x > right_limit:
                    direction = "RIGHT"
                else:
                    direction = "CENTER"

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

                overlay_boxes.append((int(x1), int(y2), label, direction, distance, int(box_height)))

        # ── Decide once, on the winner ──
        # An empty frame falls through to SAFE, so walking out of view
        # always produces an all-clear rather than leaving the cane latched.


        # 
        if top_info:
           
            top_label, top_direction, top_distance = top_info
            action = cane_decision(top_label, top_direction, top_distance)
            # Substituted with:
            # action = cane_decision(*top_info)
        # 
        
        
        
        
        # 
        else:
            # Code removed: 
            top_label, top_direction, top_distance = None, None, None
            # action = "SAFE"
            # Substituted with:           
            action = "SAFE"
        # 



        command = generate_command(action)




        # 
        
        # Entirely changed section:         # Hardware command
        #         command = generate_command(action)

        #         current_time = time.time()

        #         should_send = (
        #             command != previous_command
        #             or (current_time - last_time_sent) >= heartbeat_interval
        #         )

        #         if should_send:
        #             if command != previous_command:
        #                 print(
        #                     f"""
        # Object: {label}
        # Direction: {direction}
        # Distance: {distance}
        # Action: {action}
        # Command: {command}
        
        # send_command(command)

        # previous_command = command
        # last_time_sent = current_time
        
        # """
        # ── Stabilise, then transmit ──
        # Returns None on frames where nothing should go out.
        
        to_send = stabilizer.update(command)

        if to_send:
            if stabilizer.just_changed:
                # Full report on a real state change (not on heartbeats).
                # NOTE: hysteresis can delay a commit by a frame or two,
                # so this describes the frame the commit landed on rather
                # than necessarily the frame that began the escalation.
                print(
                    f"""
                    Object:    {top_label}
                    Direction: {top_direction}
                    Distance:  {top_distance}
                    Action:    {action}
                    Command:   {to_send}"""
                )
            send_command(to_send)

        # 
                
        

        # ── Display ──
    
        annotated = results[0].plot()


        # 
        
        # Entirely Different Section: 
        #   for (x1, y1, y2, ov_label, ov_direction, ov_distance, box_height) in overlay_boxes:
        # text = f"h={box_height} {ov_distance}"
        # text_y = min(y2 + 25, height - 10)  # keep text on-screen even for tall boxes
        # cv2.putText(
        #     annotated,
        #     text,
        #     (x1, text_y),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.7,
        #     (0, 255, 255),
        #     2,
        # )
        
        for (x, y_bottom, ov_label, ov_direction,
             ov_distance, box_h) in overlay_boxes:
            text_y = min(y_bottom + 25, height - 10)
            cv2.putText(annotated,
                        f"h={box_h} {ov_distance} {ov_direction}",
                        (x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
 
        cv2.putText(annotated, f"state: {stabilizer.state}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # 
        
        
        
        
        

        cv2.imshow("Smart Cane Vision", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    print("\nStopping vision system & turning off actuators...")
    try:
        send_command("S")
    except Exception as e:
        print("Failed to send stop command:", e)

    cap.release()
    cv2.destroyAllWindows()
# Haptic feedback controller for the smart cane (runs on the Pico).
# Save this as main.py on the Pico (via Thonny) so it auto-runs on boot.
#
# Receives a single hazard code from the host over serial:
#   S = safe          -> motor off
#   M = medium hazard  -> medium pulse (MEDIUM-distance object, any direction)
#   C = close hazard    -> fast pulse (CLOSE-distance object, any direction)
#
# The host (main.py) already resolves multiple simultaneous hazards down
# to a single "most urgent" code before sending.
# below is kept purely as a safety net in case a line ever contains more
# than one code (e.g. "M,C") -- it just picks the higher-priority one.

import select
import sys
import time
from machine import Pin

motor = Pin(15, Pin.OUT)
motor.value(0)

# Current state of the cane
current_state = "S"

# Pattern timing tracking
last_toggle_time = time.ticks_ms()
motor_on = False

# Priority order, higher number = more urgent.
PRIORITY_RANK = {
    "C": 3,  # Close hazard - high intensity
    "M": 2,  # Medium-distance hazard - medium intensity
    "S": 1,  # Safe
}


def resolve_top_priority(raw_input_string):
    """
    Takes incoming text like 'M,C' or 'C' and returns
    only the highest priority command found in it.
    """
    winning_command = "S"
    highest_score = 0

    for cmd in raw_input_string.split(","):
        clean_cmd = cmd.strip().upper()
        score = PRIORITY_RANK.get(clean_cmd, 0)
        if score > highest_score:
            highest_score = score
            winning_command = clean_cmd

    return winning_command


print("Pico Haptic Engine Ready")

try:
    while True:
        # ==========================================
        # 1. Read incoming command, non-blocking
        # ==========================================
        if sys.stdin in select.select([sys.stdin], [], [], 0.001)[0]:
            raw_line = sys.stdin.readline().strip()

            if raw_line:
                command = resolve_top_priority(raw_line)
                if command != current_state:
                    current_state = command
                    motor_on = False
                    motor.value(0)
                    last_toggle_time = time.ticks_ms()
                    print(f"Received: [{raw_line}] -> State: {current_state}")

        # ==========================================
        # 2. Haptic Pulse Generator Logic
        # ==========================================
        now = time.ticks_ms()

        # --- SAFE STATE ---
        if current_state == "S":
            motor.value(0)
            motor_on = False

        # --- MEDIUM HAZARD (medium-distance object, any direction) ---
        elif current_state == "M":
            interval = 150 if motor_on else 350
            if time.ticks_diff(now, last_toggle_time) >= interval:
                motor_on = not motor_on
                motor.value(1 if motor_on else 0)
                last_toggle_time = now

        # --- CLOSE HAZARD (close-distance object, any direction) ---
        elif current_state == "C":
            interval = 70
            if time.ticks_diff(now, last_toggle_time) >= interval:
                motor_on = not motor_on
                motor.value(1 if motor_on else 0)
                last_toggle_time = now

        time.sleep_ms(5)

except Exception as e:
    print("Error or interrupted:", e)
finally:
    motor.value(0)
    print("Motor safely turned OFF on exit")

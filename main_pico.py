"""
Smart Cane — Pico 2 W firmware.

MicroPython auto-runs main.py at power-on, so once saved the cane works
standalone: plug in a power bank and it joins WiFi, prints its IP, and
listens for commands.

FEEDBACK DESIGN
    Two channels, two independent dimensions:

        vibration  urgency only    (how bad)
        speech     direction only  (which way)

    The host sends both in one code so they can never fall out of step.

COMMAND VOCABULARY (UDP port 5005)
    Two characters: urgency then direction.

        char 0   M = medium, C = close
        char 1   L = left, C = centre, R = right

        S        safe -- the one single-character code

    Full set:  S, ML, MC, MR, CL, CC, CR

    The Pico interprets nothing. It reads char 0 to pick a vibration
    pattern and char 1 to pick a clip. Two independent lookups.
    Keep this in sync with command.py on the host.

WIRING
    Motor         -> GP15 (physical pin 20)
    DFPlayer RX   -> GP0  (physical pin 1)
    DFPlayer VCC  -> VBUS (physical pin 40)
    DFPlayer GND  -> GND  (physical pin 38)
    Speaker       -> DFPlayer SPK_1 / SPK_2 (not to the Pico)
"""

from machine import Pin, UART # type: ignore
import network # type: ignore
import socket
import time




# ── CONFIG ─────────────────────────────────────────────────

SSID     = "iPhone"
PASSWORD = "12345678"

PORT   = 5005
VOLUME = 10              # 0-30. Tune here, not in the tests.

VALID = ("S", "ML", "MC", "MR", "CL", "CC", "CR")

LINK_TIMEOUT_MS = 3000




# ── MOTORS ─────────────────────────────────────────────────

motor = Pin(15, Pin.OUT)
motor.value(0)




# ── WIFI ───────────────────────────────────────────────────

wlan = network.WLAN(network.STA_IF)
wlan.active(True)
wlan.connect(SSID, PASSWORD)

print("Connecting to", SSID)
for _ in range(40):                    # 20s ceiling, then carry on
    if wlan.isconnected():
        break
    time.sleep(0.5)

if wlan.isconnected():
    print("IP:", wlan.ifconfig()[0])   # <- put this in pico_connection.py
else:
    print("WiFi failed - check SSID/password and that the hotspot is 2.4GHz")




# ── UDP SOCKET ─────────────────────────────────────────────
# Non-blocking: recv() raises instead of waiting, so the pulse timing
# below keeps running between packets. One loop, two jobs.

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", PORT))
sock.setblocking(False)





# ── DFPLAYER ───────────────────────────────────────────────

uart = UART(0, baudrate=9600, tx=Pin(0), rx=Pin(1))
time.sleep(3)                          # module indexes the SD card at power-up

DF_PLAY = 0x03
DF_STOP = 0x16
DF_VOL  = 0x06


def df_send(cmd, param=0):
    """One 10-byte DFPlayer command frame."""
    body = [0xFF, 0x06, cmd, 0x00, (param >> 8) & 0xFF, param & 0xFF]
    chk = (-sum(body)) & 0xFFFF
    uart.write(bytes([0x7E] + body + [chk >> 8, chk & 0xFF, 0xEF]))
    time.sleep(0.05)


df_send(DF_VOL, VOLUME)

# Keyed by DIRECTION character, not by the whole code.
# 0001 = "left", 0002 = "right", 0003 = "ahead"
TRACKS = {"L": 1, "R": 2, "C": 3}


def play_for(code):
    """Play the clip for this code's direction, or stop if it has none."""
    if len(code) == 2 and code[1] in TRACKS:
        df_send(DF_PLAY, TRACKS[code[1]])





# ── READY ──────────────────────────────────────────────────

print("Smart Cane ready on port", PORT)

motor.value(1)
time.sleep(0.3)
motor.value(0)




# ── MAIN LOOP ──────────────────────────────────────────────

current_state    = "S"                 # full code, e.g. "CL"
current_urgency  = "S"                 # char 0 only -- what the motor follows
last_toggle_time = time.ticks_ms()
last_packet_at   = time.ticks_ms()     # for the link watchdog
motor_on         = False




try:
    while True:

        now = time.ticks_ms()          # every branch below reads this

        # ==========================================
        # 1. Read incoming command, non-blocking
        # ==========================================
        # recv() raises OSError when no packet is waiting -- that's most
        # cycles, not an error. cmd must be cleared here, or on the very
        # first pass it would be referenced before assignment.
        try:
            cmd = sock.recv(16).decode().strip().upper()
        except OSError:
            cmd = None
        
        # Any valid traffic proves the link is alive, including heartbeat
        # resends of the state we're already in. This must be updated
        # BEFORE the "did the state change" filter below, or repeated
        # heartbeats would never refresh the watchdog.
        if cmd in VALID:
            last_packet_at = now
        
        if cmd in VALID and cmd != current_state:
            current_state = cmd

            # ── AUDIO: reacts to every state change ──
            play_for(cmd)

            # ── VIBRATION: reacts to urgency changes only ──
            # A direction-only change (ML -> MR) leaves the motor alone:
            # same urgency, so the buzz continues uninterrupted while
            # only the speech updates.
            urgency = cmd[0]
            if urgency != current_urgency:
                current_urgency = urgency
                motor_on = False
                motor.value(0)
                last_toggle_time = now

            print(" ", cmd)

        if (current_state != "S" and time.ticks_diff(now, last_packet_at) > LINK_TIMEOUT_MS):
            current_state = "S"
            current_urgency = "S"
            motor.value(0)
            motor_on = False
            df_send(DF_STOP)
            print("Link lost - failed safe")


        # ==========================================
        # 2. Haptic Pulse Generator Logic
        # ==========================================

        # --- SAFE STATE ---
        if current_urgency == "S":
            if motor_on:
                motor.value(0)
                motor_on = False

        # --- MEDIUM HAZARD ---
        elif current_urgency == "M":
            interval = 150 if motor_on else 350
            if time.ticks_diff(now, last_toggle_time) >= interval:
                motor_on = not motor_on
                motor.value(1 if motor_on else 0)
                last_toggle_time = now

        # --- CLOSE HAZARD ---
        elif current_urgency == "C":
            interval = 70
            if time.ticks_diff(now, last_toggle_time) >= interval:
                motor_on = not motor_on
                motor.value(1 if motor_on else 0)
                last_toggle_time = now

        time.sleep_ms(5)



except Exception as e:
    print("Error:", e)

finally:
    motor.value(0)
    df_send(DF_STOP)
    print("All actuators off")
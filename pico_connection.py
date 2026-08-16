"""
Smart Cane communication layer.

The ONLY file that knows how the laptop reaches the cane.
main.py, decision.py, and command.py never change when the
transport changes -- they just call send_command(letter).

WIRED (default):
    python main.py
    Uses USB serial, exactly as before. Berry's setup is unaffected.

WIRELESS:
    CANE_MODE=wireless python main.py
    Sends UDP to the Pico's IP. Get that IP from Thonny's console
    when the Pico boots, and put it in PICO_IP below.

Neither mode is fatal if the hardware is missing:
  - wireless: UDP is fire-and-forget, packets evaporate harmlessly
  - wired: a missing port prints a warning instead of crashing,
    so the vision pipeline still runs with no board attached
"""

import os

MODE = os.environ.get("CANE_MODE", "wireless")


# ── WIRELESS ───────────────────────────────────────────────

if MODE == "wireless":

    import socket

    PICO_IP = "172.20.10.10"          # <- from Thonny's console at Pico boot
    PORT    = 5005

    _sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_command(command):
        try:
            _sock.sendto(command.encode(), (PICO_IP, PORT))
            print("Sent (udp):", command)
        except OSError as e:
            print("Command not delivered:", e)

    print(f"Cane link: wireless -> {PICO_IP}:{PORT}")


# ── WIRED ──────────────────────────────────────────────────

else:

    import serial

    PICO_PORT = "/dev/cu.usbmodem144201"      # <- ls /dev/cu.* to confirm
    BAUD      = 115200

    _pico = None

    try:
        _pico = serial.Serial(PICO_PORT, BAUD, timeout=1)
        print(f"Cane link: wired -> {PICO_PORT}")
    except Exception as e:
        # Don't kill the vision pipeline just because no board is plugged in.
        print(f"Serial port unavailable ({e}). Commands will be printed only.")

    def send_command(command):
        if _pico is None:
            print("Sent (no link):", command)
            return
        try:
            _pico.write((command + "\n").encode())
            print("Sent (serial):", command)
        except Exception as e:
            print("Command not delivered:", e)
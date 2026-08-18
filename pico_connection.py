"""
Smart Cane communication layer.

WIRELESS (Default):
    CANE_MODE=wireless python main.py
    Sends UDP to the Pico's IP. 
    
WIRED:
    python main.py
    Uses USB serial
"""

import os

MODE = os.environ.get("CANE_MODE", "wireless")


# ── WIRELESS ───────────────────────────────────────────────

if MODE == "wireless":

    import socket

    PICO_IP = "172.20.10.10"         
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

    PICO_PORT = "/dev/cu.usbmodem144201"     
    BAUD      = 115200

    pico = None

    try:
        pico = serial.Serial(PICO_PORT, BAUD, timeout=1)
        print(f"Cane link: wired -> {PICO_PORT}")
    except Exception as e:
       
        print(f"Serial port unavailable ({e}). Commands will be printed only.")

    def send_command(command):
        if pico is None:
            print("Sent (no link):", command)
            return
        try:
            pico.write((command + "\n").encode())
            print("Sent to Pico (serial):", command)
        except Exception as e:
            print("Command not delivered:", e)
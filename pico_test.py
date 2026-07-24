import serial
import time


pico = serial.Serial(
    "/dev/cu.usbmodem144201",
    115200,
    timeout=1
)


time.sleep(2)


while True:

    command = input("Enter command (L/R/C/S): ")

    pico.write((command + "\n").encode())

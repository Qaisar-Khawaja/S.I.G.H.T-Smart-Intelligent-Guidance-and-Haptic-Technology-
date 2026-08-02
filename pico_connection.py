import serial
import time


# CHANGE THIS TO YOUR PICO PORT
pico = serial.Serial(
    "/dev/cu.usbmodem144201",
    115200,
    timeout=1
)


time.sleep(2)


def send_command(command):

    pico.write(
        (command + "\n").encode()
    )

    print("Sent to Pico:", command)

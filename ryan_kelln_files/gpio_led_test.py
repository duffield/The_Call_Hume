import RPi.GPIO as GPIO
import time

# Pin configuration
LED_PIN = 17
RESISTOR_VALUE = 330  # Ohms

# Setup
GPIO.setwarnings(True)
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)

try:
    while True:
        GPIO.output(LED_PIN, GPIO.HIGH)  # Turn on LED
        time.sleep(1)
        GPIO.output(LED_PIN, GPIO.LOW)   # Turn off LED
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    GPIO.cleanup()
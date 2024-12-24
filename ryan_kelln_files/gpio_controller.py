import RPi.GPIO as GPIO
import threading
import time
import queue

from gpiozero import PWMOutputDevice

import logging
logger = logging.getLogger(__name__)

# https://pinout.xyz/pinout/pwm
PIN_LIST = [12, 13, 18, 19]

class GPIOBaseController:
    def __init__(self, pin:int, frequency:float=1000, input_range:tuple[int]=(0, 255), min_threshold:int=1, max_threshold:int=250, virtual:bool=False):
        self.pin = pin
        self.frequency = frequency
        self.input_range = input_range
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.pwm = None
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self.run)
        self.running = False
        self.virtual = virtual
        self.current_duty_cycle = 0
        is_virtual = "VIRTUAL" if self.virtual else ""
        logger.debug(f"GPIO {is_virtual} Controller initialized for pin {self.pin} and frequency {self.frequency}")

    def run(self):
        while self.running:
            try:
                duty_cycle = self.queue.get(timeout=1)
                if duty_cycle is None:
                    #logger.warning("Received None duty_cycle value, exiting thread")
                    return
                # map input range to duty cycle (0-1)
                if duty_cycle <= self.min_threshold:
                    duty_cycle = 0.
                elif duty_cycle > self.input_range[1]:
                    duty_cycle = 1.
                else:
                    if duty_cycle < self.input_range[0]:
                        duty_cycle = self.input_range[0]
                    elif duty_cycle > self.max_threshold:
                        duty_cycle = self.input_range[1]
                    duty_cycle = (duty_cycle - self.input_range[0]) / (self.input_range[1] - self.input_range[0])

                if duty_cycle == self.current_duty_cycle:
                        continue
                if self.virtual:
                    logger.debug(f"Virtual GPIO: Pin {self.pin} set to {duty_cycle * 100.0:.1f}% duty cycle")
                else:
                    self.current_duty_cycle = duty_cycle
                    self._set_pwm(duty_cycle)
            except queue.Empty:
                continue

    def update_pwm(self, value):
        if self.running and 0 <= value:
            self.queue.put(value)

    def _start_pwm(self):
        raise NotImplementedError
    
    def _stop_pwm(self):
        raise NotImplementedError

    def _set_pwm(self, value):
        raise NotImplementedError

    def start(self):
        if self.running:
            return
        self.running = True
        self._start_pwm()
        self.thread.start()

    def stop(self):
        if not self.running:
            return
        self.running = False
        self.queue.put(None)
        self.thread.join()
        self._stop_pwm()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()



class GPIOZeroController(GPIOBaseController):
    def __init__(self, pin:int, frequency:float=100, input_range:tuple[int]=(0, 255), min_threshold:int=1, max_threshold:int=250, virtual:bool=False):
        super().__init__(pin, frequency, input_range, virtual)
        self.pwm = PWMOutputDevice(pin, frequency=frequency)

    def _start_pwm(self):
        self.pwm.value = 0

    def _stop_pwm(self):
        self.pwm.value = 0
    
    def _set_pwm(self, value):
        print(value)
        self.pwm.value = value


class GPIOController(GPIOBaseController):
    def __init__(self, pin:int, frequency:float=100, input_range:tuple[int]=(0, 255), min_threshold:int=1, max_threshold:int=250, virtual:bool=False):
        super().__init__(pin, frequency, input_range, virtual)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.OUT)
        self.pwm = GPIO.PWM(self.pin, self.frequency)

    def _start_pwm(self):
        self.pwm.start(0)

    def _stop_pwm(self):
        self.pwm.stop()
        GPIO.cleanup()

    def _set_pwm(self, value:float):
        self.pwm.ChangeDutyCycle(value * 100.0) # 1 - 100


class GPIOManager:
    def __init__(self, pins:list[int], frequency:float=100, input_range:tuple[int]=(0, 255), min_threshold:int=1, max_threshold:int=250, virtual:bool=False, controller_class=GPIOZeroController):
        self.controllers = {pin: controller_class(pin, frequency, input_range, min_threshold=min_threshold, max_threshold=max_threshold, virtual=virtual) for pin in pins}
        logger.info(f"Initialized GPIO Manager with pins: {pins}, frequency: {frequency}, input range: {input_range}, min: {min_threshold}, max: {max_threshold} virtual: {virtual}")

    def update_pwm(self, pin, value):
        if pin in self.controllers:
            self.controllers[pin].update_pwm(value)

    def update(self, values):
        if len(values) != len(self.controllers):
            raise ValueError("Number of values must match number of pins")
        #logger.debug(f"Setting PWM values {values}")
        for pin, value in zip(self.controllers.keys(), values):
            self.update_pwm(pin, value)

    def start_all(self):
        for controller in self.controllers.values():
            controller.start()

    def stop_all(self):
        for controller in self.controllers.values():
            controller.stop()
        GPIO.cleanup()

    def __enter__(self):
        self.start_all()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_all()


# Usage in audio_processor.py
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test GPIO controller")
    parser.add_argument("--info", action="store_true", help="Print available GPIO pins")
    args = parser.parse_args()

    # if --info is passed, print the available pins
    if args.info:
        print("Available GPIO pins:")
        # FIXME: TODO
        exit(0)

    import random   
    gpio_pins = [13]  # Example GPIO pins
    input_range = (0, 255) 
    gpio_manager = GPIOManager(pins=gpio_pins, input_range=input_range, frequency=100, virtual=False, controller_class=GPIOZeroController)
    try:
        with gpio_manager:
            
            # gpio_manager.update([128])
            # input(">")
            for i in range(input_range[0], input_range[1]):
                # Update PWM values for each pin
                #values = [random.randint(*input_range) for _ in gpio_pins]
                values = [i]
                print(f"Setting PWM values: {values}")
                #values = [15]
                gpio_manager.update(values)
                #input(">")
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting...")
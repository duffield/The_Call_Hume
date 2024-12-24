import serial
import serial.tools.list_ports
import time
from pynput import keyboard
import threading


class SerialConnector:
    """Manages the serial connection to the Arduino and provides callback support for pin events."""

    def __init__(self, baud_rate=9600, toggle_pin=3, input_pin=8):
        self.baud_rate = baud_rate
        self.toggle_pin_number = toggle_pin  # Avoid naming conflict
        self.input_pin = input_pin
        self.toggle_pin_state = False
        self.input_pin_state = None
        self.arduino = None
        self.callback = None  # Callback to handle input pin events

    def list_serial_ports(self):
        """List all available serial ports and return the list."""
        ports = serial.tools.list_ports.comports()
        if not ports:
            print("No serial ports found.")
            return []
        else:
            print("Available serial ports:")
            for i, port in enumerate(ports):
                print(f"  [{i}] {port.device} ({port.description})")
            return ports

    def select_serial_port(self):
        """Prompt the user to select a serial port."""
        ports = self.list_serial_ports()
        if not ports:
            return None

        while True:
            try:
                choice = int(input("Select a serial port by number: "))
                if 0 <= choice < len(ports):
                    return ports[choice].device
                else:
                    print("Invalid choice. Try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    def connect(self):
        """Connect to the Arduino over the selected serial port."""
        print("Scanning for available serial ports...")
        selected_port = self.select_serial_port()
        if not selected_port:
            print("No serial ports selected. Exiting.")
            return False

        print(f"Connecting to {selected_port} at {self.baud_rate} baud...")
        try:
            self.arduino = serial.Serial(selected_port, self.baud_rate, timeout=1)
            time.sleep(2)  # Wait for the connection to initialize
            print(f"Connected to {self.arduino.name}.")
            return True
        except serial.SerialException as e:
            print(f"Failed to connect to {selected_port}: {e}")
            return False

    def toggle_pin(self):
        """Toggle the Arduino pin state and send the command."""
        self.toggle_pin_state = not self.toggle_pin_state
        command = f"TOGGLE,{self.toggle_pin_number}\n"  # Use the corrected attribute name
        self.arduino.write(command.encode())
        print(f"Pin {self.toggle_pin_number} toggled to {'HIGH' if self.toggle_pin_state else 'LOW'}.")

    def monitor_input_state(self):
        """Continuously monitor the state of the input pin and invoke the callback if registered."""
        while True:
            try:
                command = f"READ,{self.input_pin}\n"  # Format: READ,<pin>
                self.arduino.write(command.encode())
                state = self.arduino.readline().decode().strip()

                if state.isdigit():
                    state = int(state)
                    if state != self.input_pin_state:  # Only act if the state changes
                        self.input_pin_state = state
                        print(f"Input pin {self.input_pin} changed to {'HIGH' if self.input_pin_state else 'LOW'}.")
                        # if self.callback and state == 1:  # Trigger the callback on HIGH
                            # self.callback()
                        self.callback()
                time.sleep(0.1)  # Small delay to prevent flooding
            except Exception as e:
                print(f"Error while monitoring input state: {e}")
                break

    def set_callback(self, callback):
        """Register a callback function to be triggered on input pin HIGH."""
        self.callback = callback

    def start_monitoring(self):
        """Start monitoring the input pin in a separate thread."""
        monitor_thread = threading.Thread(target=self.monitor_input_state, daemon=True)
        monitor_thread.start()

    def on_key_press(self, key):
        """Handle key press events."""
        try:
            if key.char == '1':  # Toggle pin state on '1' key
                self.toggle_pin()
        except AttributeError:
            pass  # Handle special keys gracefully

    def start_key_listener(self):
        """Start a keyboard listener for toggling the pin."""
        def on_press(key):
            self.on_key_press(key)

        def on_release(key):
            if key == keyboard.Key.esc:
                print("Exiting...")
                return False

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()


if __name__ == "__main__":
    # Initialize the serial connector
    connector = SerialConnector()

    if not connector.connect():
        exit(1)

    # Define a callback function for button press
    def button_pressed():
        print("Button press detected on pin 8!")

    # Register the callback
    connector.set_callback(button_pressed)

    # Start monitoring the input pin and keyboard listener
    connector.start_monitoring()
    connector.start_key_listener()

    try:
        while True:
            time.sleep(1)  # Keep the main thread alive
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        if connector.arduino and connector.arduino.is_open:
            connector.arduino.close()
            print("Serial connection closed.")
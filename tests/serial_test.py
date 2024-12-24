import serial
import serial.tools.list_ports
import time
from pynput import keyboard
import threading

def list_serial_ports():
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

def select_serial_port():
    """Prompt the user to select a serial port."""
    ports = list_serial_ports()
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

def toggle_pin():
    """Toggle the Arduino pin state and send the command."""
    global toggle_pin_state, arduino, TOGGLE_PIN
    toggle_pin_state = not toggle_pin_state
    command = f"TOGGLE,{TOGGLE_PIN}\n"  # Format: TOGGLE,<pin>
    arduino.write(command.encode())
    print(f"Pin {TOGGLE_PIN} toggled to {'HIGH' if toggle_pin_state else 'LOW'}.")

def monitor_input_state():
    """Continuously monitor the state of the input pin and print updates."""
    global arduino, INPUT_PIN, input_pin_state
    while True:
        try:
            command = f"READ,{INPUT_PIN}\n"  # Format: READ,<pin>
            arduino.write(command.encode())
            # Wait for the Arduino to send a response
            state = arduino.readline().decode().strip()
            if state.isdigit():
                state = int(state)
                if state != input_pin_state:  # Only print if the state has changed
                    input_pin_state = state
                    print(f"Input pin {INPUT_PIN} changed to {'HIGH' if input_pin_state else 'LOW'}.")
            time.sleep(0.1)  # Small delay to prevent flooding
        except Exception as e:
            print(f"Error while monitoring input state: {e}")
            break

def on_key_press(key):
    """Handle key press events."""
    try:
        if key.char == '1':  # Toggle pin state on 'm' key
            toggle_pin()
    except AttributeError:
        pass  # Handle special keys gracefully

def on_key_release(key):
    """Handle key release events."""
    if key == keyboard.Key.esc:
        print("Exiting...")
        return False

if __name__ == "__main__":
    # Arduino pin configuration
    BAUD_RATE = 9600
    TOGGLE_PIN = 3  # Pin to toggle
    INPUT_PIN = 8   # Pin to read input state from
    toggle_pin_state = False
    input_pin_state = None  # Initialize input pin state to None

    # Select the serial port
    print("Scanning for available serial ports...")
    selected_port = select_serial_port()
    if not selected_port:
        print("No serial ports selected. Exiting.")
        exit(1)

    print(f"Connecting to {selected_port} at {BAUD_RATE} baud...")
    try:
        arduino = serial.Serial(selected_port, BAUD_RATE, timeout=1)
        time.sleep(2)  # Wait for the connection to initialize
        print(f"Connected to {arduino.name}.")
        print("Press 'm' to toggle the pin state. Press 'Esc' to exit.")

        # Start a thread to monitor the input pin state
        monitor_thread = threading.Thread(target=monitor_input_state, daemon=True)
        monitor_thread.start()

        # Set up the key listener
        with keyboard.Listener(on_press=on_key_press, on_release=on_key_release) as listener:
            listener.join()

    except serial.SerialException as e:
        print(f"Failed to connect to {selected_port}: {e}")
    finally:
        # Close the serial connection on exit
        if 'arduino' in locals() and arduino.is_open:
            arduino.close()
            print(f"Serial port {arduino.name} closed.")

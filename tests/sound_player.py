from pynput import keyboard
from playsound import playsound
import threading

# Path to your audio file
AUDIO_FILE = "1.wav"

def play_audio():
    """Function to play the audio."""
    try:
        playsound(AUDIO_FILE)
    except Exception as e:
        print(f"Error playing sound: {e}")

def on_press(key):
    """Handle key press events."""
    try:
        if key.char == "m":  # If the "1" key is pressed
            print("Playing audio...")
            # Play audio in a separate thread to avoid blocking key detection
            threading.Thread(target=play_audio).start()
    except AttributeError:
        pass

def on_release(key):
    """Handle key release events."""
    if key == keyboard.Key.esc:
        # Stop listener if 'esc' is pressed
        print("Exiting...")
        return False

# Set up the key listener
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    print("Press '1' to play the audio. Press 'Esc' to exit.")
    listener.join()

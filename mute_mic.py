import pyaudio
import numpy as np

# PyAudio parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024  # Buffer size

print("Mute Mic Running")

def callback(in_data, frame_count, time_info, status):
    # Replace audio input with silence
    silent_data = np.zeros(CHUNK, dtype=np.int16).tobytes()
    return (silent_data, pyaudio.paContinue)

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open a stream to mute the microphone
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=callback)

print("Microphone muted. Press Ctrl+C to stop.")
try:
    stream.start_stream()
    while stream.is_active():
        pass
except KeyboardInterrupt:
    print("Stopping...")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()

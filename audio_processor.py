import argparse
import threading
import time
import importlib
from pynput import keyboard
from sfx_play import SFXPlayer
from serial_connect import SerialConnector
import pygame
from logging_config import setup_logging
logger = setup_logging()

from audio_stream import (
    DEFAULT_SAMPLE_RATE, AudioStream, MicAudioStream,
    PyAudioFileAudioStream, HumeAudioStream, VoiceActivityDetector
)
from translators import get_translators

def load_components():
    logger.info("Loading translators...")
    translators = get_translators()
    for translator in translators.values():
        logger.info(f"Loading translator: {translator.__name__}")
        importlib.import_module(translator.__module__, translator.__name__)

class AudioProcessor:
    def __init__(self, audio_stream, serial_connector, sampling_rate=DEFAULT_SAMPLE_RATE, fps=30.0):
        self.audio_stream = audio_stream
        self.serial_connector = serial_connector
        self.sampling_rate = sampling_rate
        self.microphone_muted = False
        self.delay = 1.0 / fps
        self.is_playing = False
        self.sfx_player = SFXPlayer(audio_file="phone_ring.mp3", initial_delay=0.0, delay_between_plays=6.0)

        if serial_connector:
            self.serial_connector.set_callback(self.pin_state_changed)

    def pin_state_changed(self):
        if self.serial_connector.input_pin_state == 1:
            self.mute_microphone()
            if not self.is_playing:
                self.is_playing = True
                threading.Thread(target=self.handle_repeated_playback, daemon=True).start()
        elif self.serial_connector.input_pin_state == 0:
            self.unmute_microphone()
            self.is_playing = False
            self.sfx_player.stop()

    def handle_repeated_playback(self):
        while self.is_playing:
            self.sfx_player.play()
            time.sleep(self.sfx_player.delay_between_plays)

    def run(self):
        if self.serial_connector:
            threading.Thread(target=self.serial_connector.start_monitoring, daemon=True).start()
        threading.Thread(target=self.start_keyboard_listener, daemon=True).start()

        try:
            with self.audio_stream:
                while self.audio_stream.running:
                    time.sleep(0.1)
        except Exception as e:
            logger.error(f"An error occurred: {e}")

    def mute_microphone(self):
        if not self.microphone_muted:
            logger.info("Microphone muted.")
            self.microphone_muted = True
            if hasattr(self.audio_stream, 'mute_microphone'):
                self.audio_stream.mute_microphone()

    def unmute_microphone(self):
        if self.microphone_muted:
            logger.info("Microphone unmuted.")
            self.microphone_muted = False
            if hasattr(self.audio_stream, 'unmute_microphone'):
                self.audio_stream.unmute_microphone()

    def start_keyboard_listener(self):
        def on_key_press(key):
            try:
                if key.char == 'm':
                    if self.microphone_muted:
                        self.unmute_microphone()
                    else:
                        self.mute_microphone()
            except AttributeError:
                pass

        with keyboard.Listener(on_press=on_key_press) as listener:
            listener.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Processor")
    parser.add_argument("--audio", type=str, help="Path to an audio file (.wav, .mp3) or `hume` to process with Hume.ai")
    parser.add_argument("--rate", type=int, default=DEFAULT_SAMPLE_RATE, help="Sampling rate for audio stream.")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--device", type=int, default=-1)
    args = parser.parse_args()

    if args.list_devices:
        import sounddevice
        print(sounddevice.query_devices())
        exit(0)

    load_components()

    serial_connector = None
    if SerialConnector:
        serial_connector = SerialConnector()
        if not serial_connector.connect():
            logger.warning("Failed to connect to Arduino. Continuing without it.")
            serial_connector = None

    if args.audio:
        if args.audio.startswith("hume"):
            audio_stream = HumeAudioStream(device=args.device, channels=2)  # Two-channel output
        elif args.audio == "mic":
            audio_stream = MicAudioStream(device=args.device)
        else:
            audio_stream = PyAudioFileAudioStream(args.audio, rate=args.rate, loop=True)
    else:
        audio_stream = MicAudioStream(device=args.device)

    processor = AudioProcessor(audio_stream, serial_connector, sampling_rate=args.rate)
    processor.run()

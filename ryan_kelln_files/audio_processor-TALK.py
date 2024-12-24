import argparse
import time
import sys
import json
import os
import importlib
from pynput import keyboard
import threading

# Initialize logging first
from logging_config import setup_logging
logger = setup_logging()

from audio_stream import (
    DEFAULT_SAMPLE_RATE, AudioStream, MicAudioStream, 
    PyAudioFileAudioStream, HumeAudioStream, VoiceActivityDetector
)
from translators import get_translators, AlgorithmChain

"""
Audio Processor Module
This module provides classes and functions for processing audio streams and applying filters.
"""

def load_components():
    logger.info("Loading translators...")
    translators = get_translators()
    for translator in translators.values():
        logger.info(f"Loading translator: {str(translator.__name__)}")
        importlib.import_module(translator.__module__, translator.__name__)

class ParameterInput:
    def __init__(self):
        self.parameters = {'sentiment': 0.5}  # Default sentiment value (0.0 sad, 1.0 happy)

    def get_parameters(self):
        return self.parameters

    def set_sentiment(self, value):
        self.parameters['sentiment'] = value


def high_pass_filter(audio_chunk, cutoff: float = 100, fs: float = 44100):
    from scipy.signal import butter, sosfilt
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    sos = butter(N=5, Wn=normal_cutoff, btype='highpass', output='sos')
    filtered_audio = sosfilt(sos, audio_chunk)
    return filtered_audio


class AudioProcessor:
    def __init__(self, audio_stream, sampling_rate=DEFAULT_SAMPLE_RATE, num_outputs=5, use_vad=False, high_pass_filter=False, chain=None, fps=30.0):
        self.num_outputs = num_outputs
        self.audio_stream = audio_stream
        self.sampling_rate = sampling_rate
        self.parameter_input = ParameterInput()
        self.algorithm_chain = AlgorithmChain()
        self.setup_algorithms(chain)

        self.use_vad = use_vad
        self.high_pass_filter = high_pass_filter
        self.microphone_muted = False  # Track mute state

        if self.use_vad:
            self.voice_activity_detector = VoiceActivityDetector()

        self.delay = 1.0 / fps

    def setup_algorithms(self, chain_dict: dict):
        parameters = self.parameter_input.get_parameters()
        if chain_dict is None:
            chain_dict = {'frequency': 1.0}

        for translator, data in chain_dict.items():
            if translator in get_translators():
                translator_class = get_translators()[translator]
                translator_params = parameters.copy()
                if isinstance(data, float):
                    weight = data
                elif isinstance(data, dict):
                    weight = data.get('weight', 1.0)
                    translator_params.update(data)
                else:
                    raise ValueError("Invalid data format for translator.")
                algorithm = translator_class(self.num_outputs, parameters=translator_params)
                self.algorithm_chain.add_algorithm(algorithm, weight=weight)
            else:
                print("Invalid translator: ", translator)

        self.algorithm_chain.update_weights()  # normalize weights to sum to 1.0

    def run(self):
        chunk_duration = self.audio_stream.get_chunk_duration()

        # Start the keyboard listener in a separate thread
        listener_thread = threading.Thread(target=self.start_keyboard_listener, daemon=True)
        listener_thread.start()

        try:
            with self.audio_stream:
                next_process_time = time.monotonic()
                while self.audio_stream.running:
                    t = time.monotonic()
                    audio_chunk = self.audio_stream.get_audio_chunk()

                    if audio_chunk is not None and not self.microphone_muted:  # Skip processing if muted
                        if self.high_pass_filter:
                            filtered_chunk = high_pass_filter(audio_chunk, fs=self.sampling_rate)
                        else:
                            filtered_chunk = audio_chunk

                        if self.use_vad:
                            outputs = self.algorithm_chain.process(filtered_chunk, is_speech=self.voice_activity_detector.is_speech(filtered_chunk))
                        else:
                            outputs = self.algorithm_chain.process(filtered_chunk, is_speech=True)

                        # Do something with `outputs`

                        next_process_time += chunk_duration
                        now = time.monotonic()
                        delay = next_process_time - now
                        if delay < 0:
                            next_process_time = now
                    else:
                        delay = max(0.01, self.delay - (time.monotonic() - t))

                    if delay > 0:
                        time.sleep(delay)
        except KeyboardInterrupt:
            logger.info("Stream stopped by user.")
        except Exception as e:
            logger.error(f"An error occurred: {e}")

    def mute_microphone(self):
        logger.info("Microphone muted.")
        self.microphone_muted = True
        if hasattr(self.audio_stream, 'mute_microphone'):
            self.audio_stream.mute_microphone()

    def unmute_microphone(self):
        logger.info("Microphone unmuted.")
        self.microphone_muted = False
        if hasattr(self.audio_stream, 'unmute_microphone'):
            self.audio_stream.unmute_microphone()

    def toggle_microphone(self):
        if self.microphone_muted:
            self.unmute_microphone()
        else:
            self.mute_microphone()

    def start_keyboard_listener(self):
        def on_key_press(key):
            try:
                if key.char == 'm':  # Toggle mute state with 'm' key
                    self.toggle_microphone()
            except AttributeError:
                pass  # Handle special keys gracefully

        with keyboard.Listener(on_press=on_key_press) as listener:
            listener.join()


def load_chain_settings(config_path):
    with open(config_path, 'r') as file:
        chain = json.load(file)
    return chain


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Processor")
    parser.add_argument("-n", "--num_outputs", "--num-outputs", type=int, default=4, help="Number of output values")
    parser.add_argument("--vad", action="store_true", help="Use Voice Activity Detection")
    parser.add_argument("--filter", action="store_true", help="Apply high-pass filter")
    parser.add_argument("--chain", type=str, help="Path to the chain configuration file.")
    parser.add_argument("--audio", type=str, help="Path to an audio file (.wav, .mp3) for testing or `hume` to process with hume.ai")
    parser.add_argument("-r", "--rate", "--sampling_rate", type=int, default=DEFAULT_SAMPLE_RATE, help="Sampling rate for audio stream.")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--device", type=int, default=-1)
    args = parser.parse_args()

    if args.list_devices:
        import sounddevice
        print(sounddevice.query_devices())
        exit(0)

    load_components()

    chain = {'volume_random': 1.0}

    if args.chain:
        if not args.chain.endswith('.json'):
            logger.error("Chain configuration file must be a JSON file.")
            sys.exit(1)
        if not os.path.exists(args.chain):
            logger.error("Chain configuration file not found.")
            sys.exit(1)
        chain = load_chain_settings(args.chain)
    
    logger.info("Chain settings:")
    logger.info(chain)

    if args.audio:
        if args.audio.startswith("hume"):
            audio_stream = HumeAudioStream(device=args.device)
        elif args.audio == "mic":
            audio_stream = MicAudioStream(device=args.device)
        else:
            audio_stream = PyAudioFileAudioStream(args.audio, rate=args.rate, loop=True)
    else:
        logger.debug(f"Using default audio stream, microphone input with device: {args.device}")
        audio_stream = MicAudioStream(device=args.device)

    processor = AudioProcessor(audio_stream, sampling_rate=args.rate, num_outputs=args.num_outputs, use_vad=args.vad, high_pass_filter=args.filter, chain=chain, fps=30)
    processor.run()
    
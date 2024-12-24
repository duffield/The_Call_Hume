import pygame
import time
from threading import Lock

class SFXPlayer:
    def __init__(self, audio_file, initial_delay=2.0, delay_between_plays=2.0):
        """
        Initialize the sound effect player.

        :param audio_file: Path to the audio file to play.
        :param initial_delay: Delay before the first play (in seconds).
        :param delay_between_plays: Minimum delay between retriggers (in seconds).
        """
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        self.sound = pygame.mixer.Sound(audio_file)
        self.sound.set_volume(0.5)
        self.channel = pygame.mixer.Channel(0)  # Always use channel 0 for this player
        self.initial_delay = initial_delay
        self.delay_between_plays = delay_between_plays
        self.last_play_time = None  # Track the last time the sound was played
        self.lock = Lock()

    def play(self):
        """
        Play the audio file with initial and subsequent delays handled correctly.
        """
        with self.lock:
            current_time = time.time()
            if self.last_play_time is None:  # First play, apply initial delay
                print(f"Waiting for initial delay: {self.initial_delay} seconds")
                time.sleep(self.initial_delay)
                self.last_play_time = current_time
                self.channel.play(self.sound)
                print("Sound played after initial delay")
            elif current_time - self.last_play_time >= self.delay_between_plays:  # Subsequent plays
                self.channel.play(self.sound)
                self.last_play_time = current_time
                print(f"Sound played after delay of {self.delay_between_plays} seconds")

    def stop(self):
        """
        Stop the audio playback.
        """
        with self.lock:
            if self.channel.get_busy():  # Only stop if something is playing
                self.channel.stop()
                self.last_play_time = None  # Reset play tracking

"""Abstraction for handling microphone input."""

from dataclasses import dataclass
import asyncio
import contextlib
import dataclasses
import logging
from typing import AsyncIterator, ClassVar, Iterator, Optional, Callable

from hume import Stream
from hume import MicrophoneInterface as HumeMicrophoneInterface
from hume.empathic_voice.chat.audio.chat_client import ChatClient
from hume.empathic_voice.chat.audio.microphone import Microphone as HumeMicrophone

from hume.empathic_voice.chat.audio.microphone_sender import MicrophoneSender
from hume.empathic_voice.chat.socket_client import ChatWebsocketConnection
from hume.empathic_voice.chat.audio.chat_client import ChatClient
from hume.empathic_voice.types import AudioConfiguration, SessionSettings
from hume.empathic_voice.chat.audio.asyncio_utilities import Stream

from hume.core.api_error import ApiError

try:
    import _cffi_backend as cffi_backend
    import sounddevice
    from _cffi_backend import _CDataBase as CDataBase  # pylint: disable=no-name-in-module
    from sounddevice import CallbackFlags, RawInputStream

    HAS_AUDIO_DEPENDENCIES = True
except ModuleNotFoundError:
    HAS_AUDIO_DEPENDENCIES = False


logger = logging.getLogger(__name__)

# audio callback type
AudioCallbackType = Optional[Callable[[bytes], None]]

@dataclasses.dataclass
class Microphone:
    """Abstraction for handling microphone input."""

    # NOTE: use int16 for compatibility with deepgram
    DATA_TYPE: ClassVar[str] = "int16"
    DEFAULT_DEVICE: ClassVar[Optional[int]] = None
    DEFAULT_CHUNK_SIZE: ClassVar[int] = 1024

    stream: Stream[bytes]
    num_channels: int
    sample_rate: int
    chunk_size: int = DEFAULT_CHUNK_SIZE
    sample_width: int = 2  # 16-bit audio

    # NOTE: implementation based on
    # [https://python-sounddevice.readthedocs.io/en/0.4.6/examples.html#creating-an-asyncio-generator-for-audio-blocks]
    @classmethod
    @contextlib.contextmanager
    def context(cls, *, device: Optional[int] = DEFAULT_DEVICE, 
                # overrides for autodetected values
                num_channels: Optional[int] = None, sample_rate:Optional[int] = None, chunk_size: Optional[int] = None,
                mute_event:asyncio.Event=None) -> Iterator["Microphone"]:
        """Create a new microphone context.

        Args:
            device (Optional[int]): Input device ID.
        """
        if not HAS_AUDIO_DEPENDENCIES:
            raise ValueError(
                'Run `pip install "hume[microphone]"` to install dependencies required to use microphone playback.'
            )

        if device is None:
            device = sounddevice.default.device[0]
        logger.info(f"device: {device}")

        sound_device = sounddevice.query_devices(device=device)
        logger.info(f"sound_device: {sound_device}")

        if num_channels is None:
            num_channels = sound_device["max_input_channels"]

        if num_channels == 0:
            devices = sounddevice.query_devices()
            message = (
                "Selected input device does not have any input channels. \n"
                "Please set MicrophoneInterface(device=<YOUR DEVICE ID>). \n"
                f"Devices:\n{devices}"
            )
            raise IOError(message)

        if sample_rate is None:
            sample_rate = int(sound_device["default_samplerate"])

        if chunk_size is None:
            chunk_size = cls.DEFAULT_CHUNK_SIZE

        # NOTE: use asyncio.get_running_loop() over asyncio.get_event_loop() per
        # [https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.get_event_loop]
        microphone = cls(stream=Stream.new(), num_channels=num_channels, sample_rate=sample_rate, chunk_size=chunk_size)
        event_loop = asyncio.get_running_loop()

        print(f"Mic info: channels: {microphone.num_channels}, sample rate: {microphone.sample_rate}, chunk size: {microphone.chunk_size}, sample width: {microphone.sample_width}")

        stop_event = asyncio.Event()

        # pylint: disable=c-extension-no-member
        # NOTE:
        # - cffi types determined by logging; see more at [https://cffi.readthedocs.io/en/stable/ref.html]
        # - put_nowait(indata[:]) seems to block, so use call_soon_threadsafe() like the reference implementation
        def callback(indata: cffi_backend.buffer, _frames: int, _time: CDataBase, _status: CallbackFlags) -> None:
            if stop_event.is_set():
                return
            if mute_event is not None and mute_event.is_set():
                return
            event_loop.call_soon_threadsafe(microphone.stream.queue.put_nowait, indata[:])

        # Include blocksize, channels, and sample_rate
        with RawInputStream(
            callback=callback,
            dtype=cls.DATA_TYPE,
            blocksize=chunk_size,
            channels=num_channels,
            samplerate=sample_rate,
            device=device,
        ):
            try:
                yield microphone
            finally:
                stop_event.set()

    def __aiter__(self) -> AsyncIterator[bytes]:
        """Iterate over bytes of microphone input."""
        return self.stream

@dataclass
class MicrophoneInterface:
    """Interface for connecting a device microphone and user-defined audio stream to an EVI connection."""

    DEFAULT_ALLOW_USER_INTERRUPT: ClassVar[bool] = False

    @classmethod
    async def start(
        cls,
        socket: ChatWebsocketConnection,
        byte_stream: Stream[bytes],
        device: int | None = HumeMicrophone.DEFAULT_DEVICE,
        allow_user_interrupt: bool = DEFAULT_ALLOW_USER_INTERRUPT,
        # overrides for autodetected values
        num_channels: Optional[int] = None, sample_rate:Optional[int] = None, chunk_size: Optional[int] = None,
        mute_event:asyncio.Event=None
    ) -> None:
        """Start the microphone interface.

        Args:
            socket (AsyncChatWSSConnection): EVI socket.
            device (int | None): Device index for the microphone.
            allow_user_interrupt (bool): Whether to allow the user to interrupt EVI. If False, the user's microphone input is stopped from flowing to the WebSocket when audio from the assistant is playing.
            byte_stream (Stream[bytes]): Byte stream of audio data.
        """

        with Microphone.context(device=device, 
                                num_channels=num_channels, sample_rate=sample_rate, chunk_size=chunk_size,
                                mute_event=mute_event) as microphone:
            audio_config = AudioConfiguration(sample_rate=microphone.sample_rate,
                                              channels=microphone.num_channels,
                                              encoding="linear16")
            
            sender = MicrophoneSender.new(microphone=microphone, allow_interrupt=allow_user_interrupt)
            chat_client = ChatClient.new(sender=sender, byte_strs=byte_stream)
            print("Configuring socket with microphone settings...")
            session_settings_config = SessionSettings(audio=audio_config)
            await socket.send_session_settings(
                message=session_settings_config
            )
            print("Microphone connected. Say something!")
            await chat_client.run(socket=socket)
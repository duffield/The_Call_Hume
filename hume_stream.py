import os
import asyncio
import argparse
import logging
from typing import Optional
import base64

from hume.client import AsyncHumeClient
from hume.empathic_voice.chat.socket_client import ChatConnectOptions, ChatWebsocketConnection
from hume.empathic_voice.chat.types import SubscribeEvent
from hume.empathic_voice.types import UserInput
from hume.core.api_error import ApiError
from hume import Stream

#from hume_callback_client import CallbackMicrophoneInterface, AudioCallbackType
from websocket_handler import WebSocketHandler, AudioCallbackType
from hume_microphone import MicrophoneInterface

from dotenv import load_dotenv
load_dotenv()

# See: https://github.com/HumeAI/hume-api-examples/tree/main/evi-python-example

# load config id from environment variables
DEFAULT_HUME_CONFIG_ID = os.getenv("HUME_CONFIG_ID")

logger = logging.getLogger(__name__)

async def hume_stream(device:Optional[int] = -1, 
                      config_id:str=DEFAULT_HUME_CONFIG_ID,
                      audio_callback:AudioCallbackType=None,
                      mute_event:asyncio.Event=None) -> None:
    load_dotenv()

    # Retrieve the Hume API key from the environment variables
    HUME_API_KEY = os.getenv("HUME_API_KEY", None)
    if not HUME_API_KEY:
        raise ValueError("HUME_API_KEY not found in environment variables")
    HUME_SECRET_KEY = os.getenv("HUME_SECRET_KEY", None)
    if not HUME_SECRET_KEY:
        raise ValueError("HUME_SECRET_KEY not found in environment variables")
    if not config_id:
        print("config_id:", config_id)
        raise ValueError("Pass in explicit config ID or set HUME_CONFIG_ID in environment variables")
    
    # Connect and authenticate with Hume
    client = AsyncHumeClient(api_key=HUME_API_KEY)

    # Define options for the WebSocket connection, such as an EVI config id and a secret key for token authentication  
    options = ChatConnectOptions(config_id=config_id, secret_key=HUME_SECRET_KEY)

    websocket_handler = WebSocketHandler(audio_callback=audio_callback)

    # Open the WebSocket connection with the configuration options and the interface's handlers  
    async with client.empathic_voice.chat.connect_with_callbacks(  
        options=options,  
        on_open=websocket_handler.on_open,  
        on_message=websocket_handler.on_message,  
        on_close=websocket_handler.on_close,  
        on_error=websocket_handler.on_error,  
    ) as socket:
        
        websocket_handler.set_socket(socket)

        try:
            await MicrophoneInterface.start(
                socket,
                byte_stream=websocket_handler.byte_strs,
                device=device,
                allow_user_interrupt=True,
                mute_event=mute_event,
            )
        except asyncio.CancelledError:
            logger.info("hume_stream was cancelled. Cleaning up...")
            raise
        except ApiError as e:
            logger.error(f"An API error occurred in hume_stream: {e}")
            raise
        except Exception as e:
            logger.error(f"An error occurred in hume_stream: {e}")
            raise

    # try:
    #     # Start streaming EVI over your device's microphone and speakers
    #     async with client.empathic_voice.chat.connect(config_id=config_id) as socket:  
    #         await CallbackMicrophoneInterface.start(
    #             socket,
    #             device=device,
    #             allow_user_interrupt=True,
    #             audio_callback=audio_callback
    #         )
    # except asyncio.CancelledError:
    #     logger.info("hume_stream was cancelled. Cleaning up...")
    #     raise
    # except Exception as e:
    #     logger.error(f"An error occurred in hume_stream: {e}")
    #     raise

async def mic_test(device:int = -1):
    from hume_microphone import Microphone

    if device < 0:
        logger.info("Using default microphone device")
        device = None

    logger.info(f"Using microphone device: {device}")
    with Microphone.context(device=device) as microphone:
        logger.info(f"Microphone sample rate: {microphone.sample_rate}")
        logger.info(f"Microphone channels: {microphone.num_channels}")
        async for audio_chunk in microphone:
            nonzero = any(audio_chunk)  # Ensure the byte string is not empty
            if nonzero:
                logger.info(f"Received audio chunk: {len(audio_chunk)} bytes")
            await asyncio.sleep(0.01)



if __name__ == "__main__":
    load_dotenv()
    if not os.getenv("HUME_API_KEY"):
        raise ValueError("HUME_API_KEY not found in environment variables")
    if not os.getenv("HUME_SECRET_KEY"):
        raise ValueError("HUME_SECRET_KEY not found in environment variables")
    
    # reset logging
    #logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.setLevel(logging.DEBUG)

    # parse args for --list-deices and display sound devices
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--test-mic", "--test_mic", action="store_true")
    parser.add_argument("--config-id", type=str, default=DEFAULT_HUME_CONFIG_ID)
    parser.add_argument("--audio-callback", action="store_true")
    args = parser.parse_args()
    if args.list_devices:
        import sounddevice
        print(sounddevice.query_devices())
        exit(0)

    if args.test_mic:
        asyncio.run(mic_test(args.device))
        exit(0)

    audio_callback = None
    if args.audio_callback:
        def audio_callback(audio_chunk):
            print(f"Received audio chunk: {len(audio_chunk)} bytes")
        audio_callback = audio_callback

    asyncio.run(hume_stream(args.device, config_id=args.config_id, audio_callback=audio_callback))
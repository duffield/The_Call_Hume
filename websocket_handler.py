import asyncio
import base64
import datetime
import os
from typing import Optional, Callable

from hume.empathic_voice.chat.socket_client import ChatWebsocketConnection
from hume.empathic_voice.chat.types import SubscribeEvent
#from hume.empathic_voice.types import UserInput
from hume.core.api_error import ApiError
from hume import Stream

# Based on https://github.com/HumeAI/hume-api-examples/blob/main/evi-python-example/quickstart.py


# audio callback type
AudioCallbackType = Optional[Callable[[bytes], None]]

class WebSocketHandler:
    """Handler for containing the EVI WebSocket and associated socket handling behavior."""

    def __init__(self, audio_callback=None):
        """Construct the WebSocketHandler, initially assigning the socket to None and the byte stream to a new Stream object."""
        self.socket = None
        self.byte_strs = Stream.new()
        self.audio_callback = audio_callback

    def set_socket(self, socket: ChatWebsocketConnection):
        """Set the socket.
        
        This method assigns the provided asynchronous WebSocket connection
        to the instance variable `self.socket`. It is invoked after successfully
        establishing a connection using the client's connect method.

        Args:
            socket (ChatWebsocketConnection): EVI asynchronous WebSocket returned by the client's connect method.
        """
        self.socket = socket

    async def on_open(self):
        """Logic invoked when the WebSocket connection is opened."""
        print("WebSocket connection opened.")

    async def on_message(self, message: SubscribeEvent):
        """Callback function to handle a WebSocket message event.
        
        This asynchronous method decodes the message, determines its type, and 
        handles it accordingly. Depending on the type of message, it 
        might log metadata, handle user or assistant messages, process
        audio data, raise an error if the message type is "error", and more.

        This method interacts with the following message types to demonstrate logging output to the terminal:
        - [chat_metadata](https://dev.hume.ai/reference/empathic-voice-interface-evi/chat/chat#receive.Chat%20Metadata.type)
        - [user_message](https://dev.hume.ai/reference/empathic-voice-interface-evi/chat/chat#receive.User%20Message.type)
        - [assistant_message](https://dev.hume.ai/reference/empathic-voice-interface-evi/chat/chat#receive.Assistant%20Message.type)
        - [audio_output](https://dev.hume.ai/reference/empathic-voice-interface-evi/chat/chat#receive.Audio%20Output.type)

        Args:
            data (SubscribeEvent): This represents any type of message that is received through the EVI WebSocket, formatted in JSON. See the full list of messages in the API Reference [here](https://dev.hume.ai/reference/empathic-voice-interface-evi/chat/chat#receive).
        """

        # Create an empty dictionary to store expression inference scores
        scores = {}

        if message.type == "chat_metadata":
            message_type = message.type.upper()
            chat_id = message.chat_id
            chat_group_id = message.chat_group_id
            text = f"<{message_type}> Chat ID: {chat_id}, Chat Group ID: {chat_group_id}"
        elif message.type in ["user_message", "assistant_message"]:
            role = message.message.role.upper()
            message_text = message.message.content
            text = f"{role}: {message_text}"
            if message.from_text is False:
                scores = dict(message.models.prosody.scores)
        elif message.type == "audio_output":
            message_str: str = message.data
            message_bytes = base64.b64decode(message_str.encode("utf-8"))
            if self.audio_callback is not None:
                self.audio_callback(message_bytes)
            else:
                await self.byte_strs.put(message_bytes)
            return
        elif message.type == "error":
            error_message: str = message.message
            error_code: str = message.code
            raise ApiError(f"Error ({error_code}): {error_message}")
        else:
            message_type = message.type.upper()
            text = f"<{message_type}>"
        
        # Print the formatted message
        self._print_prompt(text)

        # Extract and print the top 3 emotions inferred from user and assistant expressions
        if len(scores) > 0:
            top_3_emotions = self._extract_top_n_emotions(scores, 3)
            self._print_emotion_scores(top_3_emotions)
            print("")
        else:
            print("")
    

    async def on_message2(message: SubscribeEvent):  
        """Callback function to handle a WebSocket message event.

        Args:  
            data (SubscribeEvent): This represents any type of message that is received through the EVI WebSocket, formatted in JSON. See the full list of messages in the API Reference [here](https://dev.hume.ai/reference/empathic-voice-interface-evi/chat/chat#receive).  
        """

        # Create an empty dictionary to store expression inference scores  
        scores = {}

        if message.type == "chat_metadata":  
            message_type = message.type.upper()  
            chat_id = message.chat_id  
            chat_group_id = message.chat_group_id  
            text = f"<{message_type}> Chat ID: {chat_id}, Chat Group ID: {chat_group_id}"  
        elif message.type in ["user_message", "assistant_message"]:  
            role = message.message.role.upper()  
            message_text = message.message.content  
            text = f"{role}: {message_text}"  
            if message.from_text is False:  
                scores = dict(message.models.prosody.scores)  
        elif message.type == "audio_output":  
            message_str: str = message.data  
            message_bytes = base64.b64decode(message_str.encode("utf-8"))  
            await self.byte_strs.put(message_bytes)  
            return  
        elif message.type == "error":  
            error_message: str = message.message  
            error_code: str = message.code  
            raise ApiError(f"Error ({error_code}): {error_message}") # ApiError is also an imported type  
        else:  
            message_type = message.type.upper()  
            text = f"<{message_type}>"

        print(text)

    async def on_close(self):
        """Logic invoked when the WebSocket connection is closed."""
        print("WebSocket connection closed.")

    async def on_error(self, error):
        """Logic invoked when an error occurs in the WebSocket connection.
        
        See the full list of errors [here](https://dev.hume.ai/docs/resources/errors).

        Args:
            error (Exception): The error that occurred during the WebSocket communication.
        """
        print(f"Error: {error}")

    def _print_prompt(self, text: str) -> None:
        """Print a formatted message with a timestamp.

        Args:
            text (str): The message text to be printed.
        """
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        now_str = now.strftime("%H:%M:%S")
        print(f"[{now_str}] {text}")

    def _extract_top_n_emotions(self, emotion_scores: dict, n: int) -> dict:
        """
        Extract the top N emotions based on confidence scores.

        Args:
            emotion_scores (dict): A dictionary of emotions and their corresponding confidence scores.
            n (int): The number of top emotions to extract.

        Returns:
            dict: A dictionary containing the top N emotions as keys and their raw scores as values.
        """
        # Convert the dictionary into a list of tuples and sort by the score in descending order
        sorted_emotions = sorted(emotion_scores.items(), key=lambda item: item[1], reverse=True)

        # Extract the top N emotions
        top_n_emotions = {emotion: score for emotion, score in sorted_emotions[:n]}

        return top_n_emotions

    def _print_emotion_scores(self, emotion_scores: dict) -> None:
        """
        Print the emotions and their scores in a formatted, single-line manner.

        Args:
            emotion_scores (dict): A dictionary of emotions and their corresponding confidence scores.
        """
        # Format the output string
        formatted_emotions = ' | '.join([f"{emotion} ({score:.2f})" for emotion, score in emotion_scores.items()])
        
        # Print the formatted string
        print(f"|{formatted_emotions}|")
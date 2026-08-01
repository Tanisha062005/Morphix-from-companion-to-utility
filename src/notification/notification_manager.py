"""
Morphix Desktop Companion Robot - Notification Manager Layer
============================================================
File: src/notification/notification_manager.py
Description:
    Provides a decoupled notification dispatching interface.
    Coordinates audio, voice, console print, and future UI/OS notifications
    using registered NotificationSender plugins.
"""

from abc import ABC, abstractmethod
import logging
import sys
from typing import Any, List

# Setup logger
logger = logging.getLogger("Morphix.Notification")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [Notification]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class NotificationSender(ABC):
    """
    Abstract interface representing a single notification destination channel.
    """

    @abstractmethod
    def send_notification(self, message: str, trigger_time_str: str, current_time_str: str) -> None:
        """
        Sends the notification payload.

        :param message: Text message of the notification.
        :param trigger_time_str: Formatted trigger timestamp.
        :param current_time_str: Formatted current timestamp.
        """
        pass


class ConsoleNotificationSender(NotificationSender):
    """
    Notification sender implementing sound playback, TTS speaking, and console printout.
    Sequence execution: Bark -> Piper (TTS) -> Console print.
    """

    def __init__(self, sound_player: Any, tts: Any, wake_sound_path: str) -> None:
        """
        Initializes ConsoleNotificationSender.

        :param sound_player: SoundPlayer instance.
        :param tts: TextToSpeech instance.
        :param wake_sound_path: Sound effect file path to play.
        """
        self.sound_player = sound_player
        self.tts = tts
        self.wake_sound_path = wake_sound_path

    def send_notification(self, message: str, trigger_time_str: str, current_time_str: str) -> None:
        """
        Triggers notification sequence in exact order: Bark -> Piper (TTS) -> Console print.
        """
        # 1. Play Bark Sound (synchronously/blocking where possible)
        logger.info("Executing sequence step 1: Play Bark.")
        try:
            self.sound_player.play_sound(self.wake_sound_path)
        except Exception as e:
            logger.error("Sound playback failed: %s", str(e))

        # 2. Piper Speaks
        logger.info("Executing sequence step 2: Piper TTS Speaks.")
        spoken_text = f"Tanisha, it's time to {message}."
        try:
            self.tts.speak(spoken_text)
        except Exception as e:
            logger.error("TTS output failed: %s", str(e))

        # 3. Print Console Notification
        logger.info("Executing sequence step 3: Print Console Notification.")
        print("\n" + "=" * 40)
        print("REMINDER")
        print(f"{message}")
        print(f"Current Time\n{current_time_str}")
        print(f"Trigger Time\n{trigger_time_str}")
        print("=" * 40 + "\n")


class NotificationManager:
    """
    Orchestration class that holds and triggers registered NotificationSender channels.
    """

    def __init__(self) -> None:
        self._senders: List[NotificationSender] = []

    def register_sender(self, sender: NotificationSender) -> None:
        """Registers a sender channel plugin."""
        self._senders.append(sender)
        logger.info("Registered notification channel: %s", sender.__class__.__name__)

    def notify(self, message: str, trigger_time_str: str, current_time_str: str) -> None:
        """
        Broadcasting routine to trigger all notification channels.
        """
        for sender in self._senders:
            try:
                sender.send_notification(message, trigger_time_str, current_time_str)
            except Exception as e:
                logger.error("Failed sending notification via %s: %s", sender.__class__.__name__, str(e))

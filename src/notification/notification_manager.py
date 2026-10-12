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


class DisplayNotificationSender(NotificationSender):
    """
    Notification channel that renders an active reminder card overlay on the OLED display.
    """

    def send_notification(self, message: str, trigger_time_str: str, current_time_str: str) -> None:
        from src.display.eye_animator import get_active_display
        disp = get_active_display()
        if disp:
            try:
                disp.show_reminder_overlay(title=message, due_time=trigger_time_str, timeout=6.0)
                logger.info("Rendered reminder notification card overlay on display.")
            except Exception as e:
                logger.error("Failed rendering display reminder notification: %s", str(e))


class ConsoleNotificationSender(NotificationSender):
    """
    Notification sender implementing sound playback, TTS speaking, display overlay, and console printout.
    Sequence execution: Display Card + Bark -> TTS Speaks -> Console print.
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
        Triggers notification sequence in exact order: Display Card + Bark -> TTS Speaks -> Console print.
        """
        # 1. Trigger Display Card Overlay (if display animator is running)
        from src.display.eye_animator import get_active_display
        disp = get_active_display()
        if disp:
            try:
                disp.show_reminder_overlay(title=message, due_time=trigger_time_str, timeout=6.0)
            except Exception as e:
                logger.warning("Display overlay notification failed: %s", str(e))

        # 2. Play Bark Sound (synchronously/blocking where possible)
        logger.info("Executing sequence step 1: Play Bark.")
        try:
            self.sound_player.play_sound(self.wake_sound_path)
        except Exception as e:
            logger.error("Sound playback failed: %s", str(e))

        # 3. TTS Speaks with language detection and dynamic user name
        logger.info("Executing sequence step 2: TTS Speaks.")
        try:
            from src.config.profile_manager import get_primary_user
            user_name = get_primary_user()
        except Exception:
            user_name = "there"

        # Check for Devanagari characters or Hindi keywords
        is_hindi = any('\u0900' <= ch <= '\u097f' for ch in message) or any(
            w in message.lower() for w in ['karo', 'karne', 'lena', 'dena', 'batao', 'jao', 'khana', 'peena']
        )
        if is_hindi:
            spoken_text = f"{user_name}, aapko {message} yaad dilana tha."
        else:
            spoken_text = f"{user_name}, it's time to {message}."

        try:
            self.tts.speak(spoken_text)
        except Exception as e:
            logger.error("TTS output failed: %s", str(e))

        # 4. Print Console Notification
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


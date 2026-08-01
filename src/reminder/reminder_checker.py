"""
Morphix Desktop Companion Robot - Background Reminder Checker Service
======================================================================
File: src/reminder/reminder_checker.py
Description:
    Runs a background service thread polling for due reminders.
    Utilizes the NotificationManager layer to broadcast notifications.
"""

from datetime import datetime
import logging
from pathlib import Path
import sys
import threading
import time
from typing import Optional

from src.audio.sound_player import SoundPlayer
from src.notification.notification_manager import NotificationManager, ConsoleNotificationSender
from src.reminder.reminder_manager import ReminderManager
from src.voice.text_to_speech import TextToSpeech

# Setup logger for ReminderChecker module
logger = logging.getLogger("Morphix.ReminderChecker")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [ReminderChecker]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class ReminderChecker:
    """
    Background daemon service that periodically checks and triggers due reminders.
    """

    def __init__(
        self,
        reminder_manager: Optional[ReminderManager] = None,
        sound_player: Optional[SoundPlayer] = None,
        tts: Optional[TextToSpeech] = None,
        check_interval: float = 2.0,  # default to checking every 2 seconds for responsiveness
        wake_sound_path: str = "assets/sounds/morphix_voice.wav",
        notification_manager: Optional[NotificationManager] = None
    ) -> None:
        """
        Initializes ReminderChecker background service.

        :param reminder_manager: ReminderManager instance.
        :param sound_player: SoundPlayer instance.
        :param tts: TextToSpeech instance.
        :param check_interval: Polling frequency in seconds (default: 2.0s).
        :param wake_sound_path: Sound effect file path to play on due reminder.
        :param notification_manager: Optional custom NotificationManager.
        """
        self.reminder_manager: ReminderManager = reminder_manager or ReminderManager()
        self.check_interval: float = check_interval
        self.wake_sound_path: str = wake_sound_path

        # If a custom NotificationManager is provided, use it
        # Otherwise, instantiate a default wrapper with the SoundPlayer & TTS engines
        if notification_manager:
            self.notification_manager = notification_manager
        else:
            self.notification_manager = NotificationManager()
            sp = sound_player or SoundPlayer()
            tts_engine = tts or TextToSpeech()
            console_sender = ConsoleNotificationSender(sp, tts_engine, wake_sound_path)
            self.notification_manager.register_sender(console_sender)

        self._running: bool = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Starts the background thread for checking reminders."""
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._run_loop, daemon=True, name="ReminderCheckerThread")
            self._thread.start()
            logger.info("ReminderChecker background service started (Polling interval: %.1fs).", self.check_interval)

    def stop(self) -> None:
        """Stops the background reminder checking thread."""
        if self._running:
            self._running = False
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=2.0)
            logger.info("ReminderChecker background service stopped.")

    def check_once(self) -> int:
        """
        Executes a single check pass over due reminders.

        :return: Number of due reminders processed in this pass.
        """
        try:
            # Query timezone-aware current local time
            now = datetime.now(self.reminder_manager.tz)
            due_list = self.reminder_manager.get_due_reminders(now)
            count = len(due_list)

            for rem in due_list:
                rem_id = rem.get("id")
                message = rem.get("message", "do your task")
                trigger_time_str = rem.get("trigger_time", "")

                logger.info("Triggering due reminder ID %s: '%s'", rem_id, message)

                # Format timestamps for readability
                current_time_str = now.strftime('%Y-%m-%d %H:%M:%S')
                try:
                    trig_dt = datetime.fromisoformat(trigger_time_str)
                    formatted_trigger = trig_dt.strftime('%Y-%m-%d %H:%M:%S')
                except Exception:
                    formatted_trigger = trigger_time_str

                # Notify channels in exact execution order
                self.notification_manager.notify(message, formatted_trigger, current_time_str)

                # Mark completed in persistent storage
                self.reminder_manager.mark_completed(rem_id)

            return count

        except Exception as e:
            logger.error("Error in ReminderChecker pass: %s", str(e))
            return 0

    def _run_loop(self) -> None:
        """Internal thread loop executing periodic reminder checks."""
        while self._running:
            self.check_once()
            # Sleep in small increments for responsive thread termination
            elapsed = 0.0
            while self._running and elapsed < self.check_interval:
                time.sleep(0.5)
                elapsed += 0.5

"""
Morphix Desktop Companion Robot - Background Reminder Checker Service
======================================================================
File: src/reminder/reminder_checker.py
Description:
    Runs a background service thread polling for due reminders every 10 seconds.
    When a reminder trigger time arrives:
      1. Plays assets/sounds/morphix_voice.wav (waits until playback completes).
      2. Piper TTS speaks: "Tanisha, it's time to <reminder_text>."
      3. Marks the reminder completed in persistent storage.
"""

import logging
from pathlib import Path
import sys
import threading
import time
from typing import Optional

from src.audio.sound_player import SoundPlayer
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
        check_interval: float = 10.0,
        wake_sound_path: str = "assets/sounds/morphix_voice.wav"
    ) -> None:
        """
        Initializes ReminderChecker background service.

        :param reminder_manager: ReminderManager instance.
        :param sound_player: SoundPlayer instance.
        :param tts: TextToSpeech instance.
        :param check_interval: Polling frequency in seconds (default: 10.0s).
        :param wake_sound_path: Sound effect file path to play on due reminder.
        """
        self.reminder_manager: ReminderManager = reminder_manager or ReminderManager()
        self.sound_player: SoundPlayer = sound_player or SoundPlayer()
        self.tts: TextToSpeech = tts or TextToSpeech()
        self.check_interval: float = check_interval
        self.wake_sound_path: str = wake_sound_path

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
            due_list = self.reminder_manager.get_due_reminders()
            count = len(due_list)

            for rem in due_list:
                rem_id = rem.get("id")
                rem_text = rem.get("text", "do your task")

                logger.info("Triggering due reminder ID %s: '%s'", rem_id, rem_text)

                # 1. Play morphix_voice.wav (wait until playback completes)
                try:
                    self.sound_player.play_sound(self.wake_sound_path)
                except Exception as ex:
                    logger.error("Sound playback failed for reminder %s: %s", rem_id, str(ex))

                # 2. Piper TTS speaks message
                spoken_message = f"Tanisha, it's time to {rem_text}."
                logger.info("Speaking reminder notification: '%s'", spoken_message)
                try:
                    self.tts.speak(spoken_message)
                except Exception as ex:
                    logger.error("TTS speaking failed for reminder %s: %s", rem_id, str(ex))

                # 3. Mark reminder completed
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

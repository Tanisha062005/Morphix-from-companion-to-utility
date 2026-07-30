"""
Morphix Desktop Companion Robot - Reminder Manager & Checker Unit Tests
========================================================================
File: tests/test_reminder.py
"""

from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.reminder.reminder_checker import ReminderChecker
from src.reminder.reminder_manager import ReminderManager, parse_time_expression


class TestReminderManager(unittest.TestCase):
    """
    Tests ReminderManager creation, time parsing, due checking, and persistent storage.
    """

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_json = Path(self.temp_dir.name) / "test_reminders.json"
        self.manager = ReminderManager(file_path=str(self.test_json))

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_parse_time_expression_relative(self) -> None:
        now = datetime(2026, 7, 29, 10, 0, 0)
        dt = parse_time_expression("in 10 minutes", now=now)
        self.assertEqual(dt, datetime(2026, 7, 29, 10, 10, 0))

        dt2 = parse_time_expression("after 1 hour 30 minutes", now=now)
        self.assertEqual(dt2, datetime(2026, 7, 29, 11, 30, 0))

    def test_parse_time_expression_absolute(self) -> None:
        now = datetime(2026, 7, 29, 10, 0, 0)
        dt = parse_time_expression("5 PM", now=now)
        self.assertEqual(dt, datetime(2026, 7, 29, 17, 0, 0))

        dt_tom = parse_time_expression("tomorrow at 9 AM", now=now)
        self.assertEqual(dt_tom, datetime(2026, 7, 30, 9, 0, 0))

    def test_add_reminder(self) -> None:
        rem = self.manager.add_reminder("Drink water", "in 10 minutes")
        self.assertIsNotNone(rem.get("id"))
        self.assertEqual(rem.get("text"), "Drink water")
        self.assertFalse(rem.get("completed"))
        self.assertEqual(rem.get("repeat"), "none")

        all_rem = self.manager.get_all_reminders()
        self.assertEqual(len(all_rem), 1)

    def test_due_reminders(self) -> None:
        # Create a reminder in the past
        past_dt = datetime.now() - timedelta(minutes=5)
        rem = self.manager.add_reminder("Take medicine", "5 PM")
        rem["trigger_time"] = past_dt.isoformat(timespec="seconds")
        self.manager._write_to_file([rem])

        due = self.manager.get_due_reminders()
        self.assertEqual(len(due), 1)
        self.assertEqual(due[0]["text"], "Take medicine")

    def test_mark_completed(self) -> None:
        rem = self.manager.add_reminder("Exercise", "in 30 minutes")
        rem_id = rem["id"]
        res = self.manager.mark_completed(rem_id)
        self.assertTrue(res)

        updated = self.manager.get_all_reminders()[0]
        self.assertTrue(updated.get("completed"))

    def test_reminder_checker_execution(self) -> None:
        mock_sound_player = MagicMock()
        mock_tts = MagicMock()

        checker = ReminderChecker(
            reminder_manager=self.manager,
            sound_player=mock_sound_player,
            tts=mock_tts,
            wake_sound_path="assets/sounds/morphix_voice.wav"
        )

        # Add due reminder
        past_dt = datetime.now() - timedelta(seconds=10)
        rem = self.manager.add_reminder("drink water", "now")
        rem["trigger_time"] = past_dt.isoformat(timespec="seconds")
        self.manager._write_to_file([rem])

        processed = checker.check_once()
        self.assertEqual(processed, 1)

        mock_sound_player.play_sound.assert_called_once_with("assets/sounds/morphix_voice.wav")
        mock_tts.speak.assert_called_once_with("Tanisha, it's time to drink water.")

        all_rems = self.manager.get_all_reminders()
        self.assertTrue(all_rems[0].get("completed"))


if __name__ == "__main__":
    unittest.main()

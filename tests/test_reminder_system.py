"""
Morphix Desktop Companion Robot - Offline Reminder System Test Suite
====================================================================
File: tests/test_reminder_system.py
Description:
    Comprehensive verification tests for the new Reminder System:
    Cleaner, parser, invalid phrases, timezone calculations, storage,
    checker, notification ordering, active mode compatibility, and performance.
"""

import os
import sys
import unittest
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path for clean import resolution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.nlp.transcript_cleaner import TranscriptCleaner
from src.intent.reminder_parser import ReminderIntentParser
from src.reminder.reminder_manager import ReminderManager, parse_time_expression
from src.reminder.reminder_checker import ReminderChecker
from src.notification.notification_manager import NotificationManager, ConsoleNotificationSender
from src.audio.sound_player import SoundPlayer
from src.voice.text_to_speech import TextToSpeech
import run_voice_assistant


class TestTranscriptCleaner(unittest.TestCase):
    """Category 1: Transcript Cleaner Tests"""

    def test_cleaner_normalization(self) -> None:
        # Test input normalization
        self.assertEqual(
            TranscriptCleaner.clean("Remind   me   after   two   minutes"),
            "remind me after 2 minutes"
        )
        self.assertEqual(
            TranscriptCleaner.clean("Hello, Morphix! Remind me to sleep at 10 P.M."),
            "hello morphix remind me to sleep at 10 p m"
        )
        self.assertEqual(
            TranscriptCleaner.clean("to do list"),
            "todo list"
        )


class TestReminderParser(unittest.TestCase):
    """Category 2: Reminder Parser Tests"""

    def test_reminder_phrases(self) -> None:
        phrases = [
            ("remind me after 2 minutes", {"intent": "SET_REMINDER", "message": "", "delay": 120, "time_expr": "after 2 minutes"}),
            ("remind me after two minutes", {"intent": "SET_REMINDER", "message": "", "delay": 120, "time_expr": "after 2 minutes"}),
            ("remind me in 5 minutes", {"intent": "SET_REMINDER", "message": "", "delay": 300, "time_expr": "in 5 minutes"}),
            ("remind me after one hour", {"intent": "SET_REMINDER", "message": "", "delay": 3600, "time_expr": "after 1 hour"}),
            ("remind me tomorrow", {"intent": "SET_REMINDER", "message": "", "time_expr": "tomorrow"}),
            ("remind me tomorrow at 5 pm", {"intent": "SET_REMINDER", "message": "", "time_expr": "tomorrow at 5 pm"}),
            ("remind me today at 8 pm", {"intent": "SET_REMINDER", "message": "", "time_expr": "today at 8 pm"}),
            ("remind me after 30 seconds", {"intent": "SET_REMINDER", "message": "", "delay": 30, "time_expr": "after 30 seconds"})
        ]
        for phrase, expected in phrases:
            with self.subTest(phrase=phrase):
                cleaned = TranscriptCleaner.clean(phrase)
                parsed = ReminderIntentParser.parse(cleaned)
                self.assertEqual(parsed.get("intent"), expected["intent"])
                if "delay" in expected:
                    self.assertEqual(parsed.get("delay"), expected["delay"])
                if "time_expr" in expected:
                    self.assertEqual(parsed.get("time_expr"), expected["time_expr"])


class TestInvalidInputs(unittest.TestCase):
    """Category 3: Invalid Input Tests"""

    def test_invalid_reminders(self) -> None:
        invalid_examples = ["remind", "after", "reminder", "hello", "how are you"]
        for ex in invalid_examples:
            with self.subTest(ex=ex):
                cleaned = TranscriptCleaner.clean(ex)
                parsed = ReminderIntentParser.parse(cleaned)
                self.assertEqual(parsed.get("intent"), "UNKNOWN")


class TestTimezoneCalculations(unittest.TestCase):
    """Category 4: Timezone Tests"""

    def test_different_timezones(self) -> None:
        timezones = ["Asia/Kolkata", "America/New_York", "Europe/London"]
        base_now_utc = datetime(2026, 8, 1, 12, 0, 0, tzinfo=ZoneInfo("UTC"))

        for tz_name in timezones:
            with self.subTest(tz=tz_name):
                tz = ZoneInfo(tz_name)
                now_local = base_now_utc.astimezone(tz)

                # Test tomorrow at 5 PM local
                target = parse_time_expression("tomorrow at 5 pm", tz, now_local)
                self.assertEqual(target.tzinfo, tz)
                self.assertEqual(target.hour, 17)
                self.assertEqual(target.minute, 0)
                self.assertEqual(target.date(), (now_local + timedelta(days=1)).date())


class TestReminderStorage(unittest.TestCase):
    """Category 5: Reminder Storage Tests"""

    def setUp(self) -> None:
        self.test_file = str(Path(__file__).parent / "temp_reminders.json")
        self.mgr = ReminderManager(file_path=self.test_file)

    def tearDown(self) -> None:
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_storage_ops(self) -> None:
        # Save
        rem = self.mgr.add_reminder("buy milk", "after 10 minutes")
        self.assertEqual(rem["message"], "buy milk")
        self.assertEqual(rem["status"], "pending")

        # Reload
        mgr2 = ReminderManager(file_path=self.test_file)
        all_rems = mgr2.get_all_reminders()
        self.assertEqual(len(all_rems), 1)
        self.assertEqual(all_rems[0]["message"], "buy milk")

        # Multiple
        self.mgr.add_reminder("clean room", "after 20 minutes")
        self.assertEqual(len(self.mgr.get_pending_reminders()), 2)

        # Completed
        self.mgr.mark_completed(rem["id"])
        self.assertEqual(self.mgr.get_all_reminders()[0]["status"], "completed")
        self.assertEqual(len(self.mgr.get_pending_reminders()), 1)

        # Cancelled
        self.mgr.cancel_reminder(all_rems[0]["id"])
        self.assertEqual(self.mgr.get_all_reminders()[0]["status"], "cancelled")


class TestReminderChecker(unittest.TestCase):
    """Category 6: Reminder Checker Tests"""

    def setUp(self) -> None:
        self.test_file = str(Path(__file__).parent / "temp_checker_reminders.json")
        self.mgr = ReminderManager(file_path=self.test_file)
        self.sound_player = MagicMock(spec=SoundPlayer)
        self.tts = MagicMock(spec=TextToSpeech)
        self.checker = ReminderChecker(
            reminder_manager=self.mgr,
            sound_player=self.sound_player,
            tts=self.tts,
            check_interval=0.1
        )

    def tearDown(self) -> None:
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_checker_trigger(self) -> None:
        # Add reminder due immediately (1 second ago)
        rem = self.mgr.add_reminder("drink water", "after 0 seconds")
        self.mgr._write_to_file([{
            **rem,
            "trigger_time": (datetime.now(self.mgr.tz) - timedelta(seconds=1)).isoformat()
        }])

        # Trigger once
        triggered = self.checker.check_once()
        self.assertEqual(triggered, 1)

        # Verify status is completed
        all_rems = self.mgr.get_all_reminders()
        self.assertEqual(all_rems[0]["status"], "completed")

        # Verify it does not trigger again
        triggered_again = self.checker.check_once()
        self.assertEqual(triggered_again, 0)


class TestNotificationExecution(unittest.TestCase):
    """Category 7: Notification Tests"""

    def test_sequence_order(self) -> None:
        mock_sp = MagicMock()
        mock_tts = MagicMock()

        # Track call sequence
        call_order = []
        mock_sp.play_sound.side_effect = lambda path: call_order.append("bark")
        mock_tts.speak.side_effect = lambda text: call_order.append("speak")

        sender = ConsoleNotificationSender(mock_sp, mock_tts, "assets/sounds/morphix_voice.wav")

        with patch('builtins.print') as mock_print:
            mock_print.side_effect = lambda *args, **kwargs: call_order.append("print")
            sender.send_notification("hello", "time1", "time2")

        # Deduplicate consecutive entries to verify sequence step order
        dedup_order = []
        for x in call_order:
            if not dedup_order or dedup_order[-1] != x:
                dedup_order.append(x)

        self.assertEqual(dedup_order, ["bark", "speak", "print"])


class TestActiveListeningCompatibility(unittest.TestCase):
    """Category 8: Active Listening Compatibility"""

    def test_does_not_interrupt_active_mode(self) -> None:
        run_voice_assistant._current_state = run_voice_assistant.AssistantState.ACTIVE_MODE

        mock_sp = MagicMock()
        mock_tts = MagicMock()
        sender = ConsoleNotificationSender(mock_sp, mock_tts, "assets/sound.wav")
        sender.send_notification("water", "t1", "t2")

        # Assistant state must remain unaffected
        self.assertEqual(run_voice_assistant._current_state, run_voice_assistant.AssistantState.ACTIVE_MODE)


class TestPerformance(unittest.TestCase):
    """Category 9: Performance Tests"""

    def setUp(self) -> None:
        self.test_file = str(Path(__file__).parent / "temp_perf_reminders.json")
        self.mgr = ReminderManager(file_path=self.test_file)

    def tearDown(self) -> None:
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_performance_100_reminders(self) -> None:
        for i in range(100):
            self.mgr.add_reminder(f"task {i}", "after 10 minutes")

        start_time = time.perf_counter()
        due = self.mgr.get_due_reminders()
        end_time = time.perf_counter()

        self.assertEqual(len(due), 0)
        self.assertLess(end_time - start_time, 0.05)  # Lookup is fast (< 50ms)


if __name__ == "__main__":
    unittest.main()

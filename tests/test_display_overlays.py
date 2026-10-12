"""
Morphix Desktop Companion Robot - Unit Tests for On-Screen UI Overlays
========================================================================
File: tests/test_display_overlays.py
Description:
    Tests To-Do and Reminder UI card overlays on EyeAnimator and EyeAnimatorThread,
    as well as integration with TodoIntentPlugin, ReminderIntentPlugin, and ReminderChecker.
"""

import os
import time
import unittest
from unittest.mock import MagicMock, patch

from src.display.eye_animator import EyeAnimator, EyeAnimatorThread, get_active_display, set_active_display
from src.intent.todo_plugin import TodoIntentPlugin, TodoService
from src.intent.reminder_plugin import ReminderIntentPlugin, ReminderService
from src.notification.notification_manager import ConsoleNotificationSender, DisplayNotificationSender, NotificationManager


class TestDisplayOverlays(unittest.TestCase):

    def setUp(self):
        self.animator = EyeAnimator(width=480, height=240)

    def test_show_todo_overlay(self):
        tasks = ["Buy groceries", "Call Mom", "Finish Morphix embedded system"]
        self.animator.show_todo_overlay(tasks, timeout=6.0)

        self.assertEqual(self.animator._card_overlay_type, "TODO")
        self.assertEqual(len(self.animator._card_todo_tasks), 3)
        self.assertEqual(self.animator._card_todo_tasks[0], "Buy groceries")
        self.assertEqual(self.animator._card_timeout, 6.0)
        self.assertEqual(self.animator._card_elapsed, 0.0)

    def test_show_reminder_overlay(self):
        title = "Water the plants"
        due_time = "17:30"
        self.animator.show_reminder_overlay(title, due_time, timeout=6.0)

        self.assertEqual(self.animator._card_overlay_type, "REMINDER")
        # show_reminder_overlay now stores in _card_reminder_list (list of dicts)
        self.assertGreater(len(self.animator._card_reminder_list), 0)
        self.assertEqual(self.animator._card_reminder_list[0]["text"], "Water the plants")
        self.assertEqual(self.animator._card_reminder_list[0]["due"], "17:30")
        self.assertEqual(self.animator._card_timeout, 6.0)

    def test_overlay_timeout(self):
        # Use 5.0s so it is above the 4.0s minimum clamp applied by show_todo_overlay
        self.animator.show_todo_overlay(["Task 1"], timeout=5.0)
        self.assertEqual(self.animator._card_overlay_type, "TODO")

        # Simulate update passes — total must exceed 5.0 s to trigger auto-dismiss
        self.animator.update(2.5)
        self.assertEqual(self.animator._card_overlay_type, "TODO")

        self.animator.update(3.0)  # Total 5.5s >= 5.0s
        self.assertIsNone(self.animator._card_overlay_type)

    def test_clear_overlay(self):
        self.animator.show_reminder_overlay("Test", "12:00")
        self.assertEqual(self.animator._card_overlay_type, "REMINDER")

        self.animator.clear_overlay()
        self.assertIsNone(self.animator._card_overlay_type)

    def test_global_display_registry(self):
        thread = MagicMock()
        thread.is_alive.return_value = True
        set_active_display(thread)

        self.assertEqual(get_active_display(), thread)

        thread.is_alive.return_value = False
        self.assertIsNone(get_active_display())
        set_active_display(None)

    def test_todo_plugin_match_show(self):
        plugin = TodoIntentPlugin()
        res = plugin.match("show my to-do list")
        self.assertIsNotNone(res)
        self.assertEqual(res.intent.value, "SHOW_TODO")

        res_read = plugin.match("what are my tasks")
        self.assertIsNotNone(res_read)
        self.assertEqual(res_read.intent.value, "SHOW_TODO")

    def test_reminder_plugin_match_show(self):
        plugin = ReminderIntentPlugin()
        res = plugin.match("show my reminders")
        self.assertIsNotNone(res)
        self.assertEqual(res.intent.value, "SHOW_REMINDERS")

        res_ask = plugin.match("did I have any reminders?")
        self.assertIsNotNone(res_ask)
        self.assertEqual(res_ask.intent.value, "SHOW_REMINDERS")

    def test_unified_set_overlay_manager(self):
        # 1. TODO_VIEW
        self.animator.set_overlay("TODO_VIEW", tasks=["Buy milk", "Review code"], timeout=6.0)
        self.assertEqual(self.animator.active_overlay, "TODO_VIEW")
        self.assertEqual(self.animator.overlay_data, ["Buy milk", "Review code"])

        # 2. FESTIVAL_GANESH
        self.animator.set_overlay("FESTIVAL_GANESH", timeout=6.0)
        self.assertEqual(self.animator.active_overlay, "FESTIVAL_GANESH")
        self.assertEqual(self.animator._festival_theme, "MODAK")

        # 3. FESTIVAL_HOLI
        self.animator.set_overlay("FESTIVAL_HOLI", timeout=6.0)
        self.assertEqual(self.animator.active_overlay, "FESTIVAL_HOLI")
        self.assertEqual(self.animator._festival_theme, "HOLI")

        # 4. FESTIVAL_INDEPENDENCE
        self.animator.set_overlay("FESTIVAL_INDEPENDENCE", timeout=6.0)
        self.assertEqual(self.animator.active_overlay, "FESTIVAL_INDEPENDENCE")
        self.assertEqual(self.animator._festival_theme, "TRICOLOR")

        # 5. Clear overlay
        self.animator.clear_overlay()
        self.assertIsNone(self.animator.active_overlay)

    def test_drawing_methods_callable(self):
        import pygame
        pygame.init()
        surf = pygame.Surface((480, 240))
        self.animator.draw_todo_overlay(surf, ["Task 1", "Task 2"])
        self.animator.draw_ganesh_festival_overlay(surf)
        self.animator.draw_holi_festival_overlay(surf)
        self.animator.draw_independence_festival_overlay(surf)
        self.animator.draw_procedural_eyes(surf)
        self.animator.draw(surf)

    def test_hinglish_tudu_intent_matching(self):
        plugin = TodoIntentPlugin()
        # 1. Bug 2 query: "bro mujhe meri tudu list dikha display per"
        res1 = plugin.match("bro mujhe meri tudu list dikha display per")
        self.assertIsNotNone(res1)
        self.assertEqual(res1.intent.value, "SHOW_TODO")

        # 2. Other tudu variations
        res2 = plugin.match("mera tudu dikhao")
        self.assertIsNotNone(res2)
        self.assertEqual(res2.intent.value, "SHOW_TODO")

        res3 = plugin.match("tudu list batao")
        self.assertIsNotNone(res3)
        self.assertEqual(res3.intent.value, "SHOW_TODO")

        res4 = plugin.match("clear tudu")
        self.assertIsNotNone(res4)
        self.assertEqual(res4.intent.value, "CLEAR_TODO")

        res5 = plugin.match("add buy bread to my tudu list")
        self.assertIsNotNone(res5)
        self.assertEqual(res5.intent.value, "ADD_TODO")
        self.assertEqual(res5.parameters.get("task"), "buy bread")

    def test_todo_plugin_handle_triggers_overlay(self):
        plugin = TodoIntentPlugin()
        mock_disp = MagicMock()
        mock_disp.is_alive.return_value = True
        with patch.object(plugin.service.todo_manager, "get_pending_tasks", return_value=[{"task": "Buy milk"}]):
            res = plugin.match("show todo")
            speech = plugin.handle(res, display=mock_disp)
            mock_disp.set_overlay.assert_called_once_with("TODO_VIEW", tasks=["Buy milk"], timeout=10.0)
            self.assertIn("Buy milk", speech)

    def test_display_notification_sender(self):
        mock_disp = MagicMock()
        with patch("src.display.eye_animator.get_active_display", return_value=mock_disp):
            sender = DisplayNotificationSender()
            sender.send_notification("Take medicine", "18:00:00", "17:59:59")
            mock_disp.show_reminder_overlay.assert_called_once_with(
                title="Take medicine",
                due_time="18:00:00",
                timeout=6.0
            )

    def test_no_internet_overlay(self):
        # Test setting NO_INTERNET overlay
        self.animator.set_overlay("NO_INTERNET", timeout=5.0)
        self.assertEqual(self.animator.active_overlay, "NO_INTERNET")

        # Test drawing NO_INTERNET overlay
        import pygame
        pygame.init()
        surf = pygame.Surface((480, 240))
        self.animator.draw_no_internet_overlay(surf)
        self.animator.draw(surf)

    def test_reminder_plugin_expanded_queries(self):
        plugin = ReminderIntentPlugin()
        queries = [
            "do I have any kind of reminders right now",
            "any reminders for me",
            "check my reminders",
            "kya koi reminder hai",
            "check the reminders",
            "kya mera koi reminder hai",
            "mere reminders batao",
            "show me my reminders",
            "what are my reminders"
        ]
        for q in queries:
            res = plugin.match(q)
            self.assertIsNotNone(res, f"Failed to match query: '{q}'")
            self.assertEqual(res.intent.value, "SHOW_REMINDERS", f"Query '{q}' classified as {res.intent.value}")

    def test_reminder_plugin_handle_triggers_overlay(self):
        plugin = ReminderIntentPlugin()
        mock_disp = MagicMock()
        mock_disp.is_alive.return_value = True
        with patch.object(plugin.service.reminder_manager, "get_pending_reminders", return_value=[{"text": "Call Mom", "trigger_time": "2026-09-27T10:00:00"}]):
            res = plugin.match("check my reminders")
            speech = plugin.handle(res, display=mock_disp)
            mock_disp.show_reminders_overlay.assert_called_once()
            self.assertIn("Call Mom", speech)


if __name__ == "__main__":
    unittest.main()


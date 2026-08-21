"""
Morphix Desktop Companion Robot - Intent Classifier Unit Tests
===============================================================
File: tests/test_intent_classifier.py
"""

import unittest
from src.intent.intent_classifier import IntentClassifier, IntentType


class TestIntentClassifier(unittest.TestCase):
    """
    Tests rule-based IntentClassifier plugin matching and parameter extraction.
    """

    def setUp(self) -> None:
        self.classifier = IntentClassifier()

    def test_add_todo_intents(self) -> None:
        examples = [
            ("Add buy milk to my to-do list", "buy milk"),
            ("Add finish robotics project", "finish robotics project"),
            ("Add call mom", "call mom"),
            ("Add task clean the room", "clean the room")
        ]
        for command, expected_task in examples:
            with self.subTest(command=command):
                res = self.classifier.classify(command)
                self.assertEqual(res.intent, IntentType.ADD_TODO)
                self.assertIn("task", res.parameters)
                self.assertEqual(res.parameters["task"].lower(), expected_task.lower())

    def test_show_todo_intents(self) -> None:
        examples = [
            "Show my to-do list",
            "Show my todo list",
            "What are my tasks",
            "Read my to-do list",
            "List my tasks"
        ]
        for command in examples:
            with self.subTest(command=command):
                res = self.classifier.classify(command)
                self.assertEqual(res.intent, IntentType.SHOW_TODO)

    def test_complete_todo_intents(self) -> None:
        examples = [
            ("I completed buy milk", "buy milk"),
            ("Mark buy milk as done", "buy milk"),
            ("Complete finish robotics project", "finish robotics project")
        ]
        for command, expected_task in examples:
            with self.subTest(command=command):
                res = self.classifier.classify(command)
                self.assertEqual(res.intent, IntentType.COMPLETE_TODO)
                self.assertIn("task", res.parameters)
                self.assertIn(expected_task.lower(), res.parameters["task"].lower())

    def test_delete_todo_intents(self) -> None:
        examples = [
            ("Delete buy milk", "buy milk"),
            ("Remove call mom", "call mom"),
            ("Delete task clean room", "clean room")
        ]
        for command, expected_task in examples:
            with self.subTest(command=command):
                res = self.classifier.classify(command)
                self.assertEqual(res.intent, IntentType.DELETE_TODO)
                self.assertIn("task", res.parameters)
                self.assertIn(expected_task.lower(), res.parameters["task"].lower())

    def test_clear_todo_intents(self) -> None:
        examples = [
            "Clear my to-do list",
            "Clear my todo list",
            "Delete all tasks",
            "Remove all tasks"
        ]
        for command in examples:
            with self.subTest(command=command):
                res = self.classifier.classify(command)
                self.assertEqual(res.intent, IntentType.CLEAR_TODO)

    def test_add_reminder_intents(self) -> None:
        examples = [
            ("Remind me to drink water at 5 PM", "drink water", "5 pm"),
            ("Remind me to take medicine tomorrow at 8 AM", "take medicine", "tomorrow at 8 am"),
            ("Remind me in 10 minutes to exercise", "exercise", "in 10 minutes"),
            ("Remind me after 1 hour 30 minutes", "remind me after 1 hour 30 minutes", "")
        ]
        for command, exp_text, exp_time in examples:
            with self.subTest(command=command):
                res = self.classifier.classify(command)
                self.assertEqual(res.intent, IntentType.ADD_REMINDER)
                self.assertIn("text", res.parameters)

    def test_show_reminders_intents(self) -> None:
        examples = [
            "Show my reminders",
            "What are my reminders",
            "Read my reminders",
            "List my reminders"
        ]
        for command in examples:
            with self.subTest(command=command):
                res = self.classifier.classify(command)
                self.assertEqual(res.intent, IntentType.SHOW_REMINDERS)

    def test_unknown_intents(self) -> None:
        unknown_commands = [
            "Can you do a backflip",
            "Sing a song",
            "Tell me a joke",
            "Fly to Mars"
        ]
        for command in unknown_commands:
            with self.subTest(command=command):
                res = self.classifier.classify(command)
                self.assertEqual(res.intent, IntentType.UNKNOWN)


if __name__ == "__main__":
    unittest.main()

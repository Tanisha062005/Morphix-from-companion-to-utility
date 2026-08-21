"""
Morphix Desktop Companion Robot - Groq Conversational Brain Test Suite
======================================================================
File: tests/test_groq_chat.py
Description:
    Unit tests for the GroqChatManager class using patches to verify
    rolling history window limits, API error fallbacks, and response parsing.
"""

import os
import unittest
from unittest.mock import MagicMock, patch
from src.ai.groq_chat import GroqChatManager


class TestGroqChatManager(unittest.TestCase):
    """Verifies history logic, initialization, and error boundaries for Groq chat."""

    def setUp(self) -> None:
        # Patch Groq client construction
        self.groq_patcher = patch("src.ai.groq_chat.Groq")
        self.mock_groq_class = self.groq_patcher.start()
        self.mock_groq_instance = MagicMock()
        self.mock_groq_class.return_value = self.mock_groq_instance
        
        # Stub API key in environment
        self.original_key = os.environ.get("GROQ_API_KEY")
        os.environ["GROQ_API_KEY"] = "mock_groq_api_key_longer_than_15"

    def tearDown(self) -> None:
        self.groq_patcher.stop()
        if self.original_key is not None:
            os.environ["GROQ_API_KEY"] = self.original_key
        else:
            os.environ.pop("GROQ_API_KEY", None)

    def test_init(self) -> None:
        mgr = GroqChatManager()
        self.assertIsNotNone(mgr.client)
        self.assertEqual(mgr.model, "openai/gpt-oss-120b")
        self.assertEqual(len(mgr.history), 0)

    def test_chat_success(self) -> None:
        mgr = GroqChatManager()
        
        # Mock completion choice response
        mock_choice = MagicMock()
        mock_choice.message.content = "Hello there! I am Morphix."
        self.mock_groq_instance.chat.completions.create.return_value.choices = [mock_choice]
        
        reply = mgr.chat("hello")
        self.assertEqual(reply, "Hello there! I am Morphix.")
        
        # Verify history was updated
        self.assertEqual(len(mgr.history), 2)
        self.assertEqual(mgr.history[0], {"role": "user", "content": "hello"})
        self.assertEqual(mgr.history[1], {"role": "assistant", "content": "Hello there! I am Morphix."})

    def test_history_window_limit(self) -> None:
        mgr = GroqChatManager()
        
        # Mock responses
        mock_choice = MagicMock()
        mock_choice.message.content = "Reply"
        self.mock_groq_instance.chat.completions.create.return_value.choices = [mock_choice]
        
        # Trigger 5 chat interactions (which would add 10 messages)
        for i in range(5):
            mgr.chat(f"input {i}")
            
        # Verify history is truncated to last 4 turns (8 messages)
        self.assertEqual(len(mgr.history), 8)
        self.assertEqual(mgr.history[-2], {"role": "user", "content": "input 4"})
        self.assertEqual(mgr.history[-1], {"role": "assistant", "content": "Reply"})

    def test_chat_api_failure(self) -> None:
        mgr = GroqChatManager()
        
        # Simulate API Exception
        self.mock_groq_instance.chat.completions.create.side_effect = Exception("Groq rate limit")
        
        reply = mgr.chat("hello")
        self.assertEqual(reply, "Sorry, I had a glitch processing that!")
        self.assertEqual(len(mgr.history), 0)

    def test_chat_with_emotion_happy(self) -> None:
        mgr = GroqChatManager()
        mock_choice = MagicMock()
        mock_choice.message.content = "[HAPPY] That sounds wonderful! I am so glad."
        self.mock_groq_instance.chat.completions.create.return_value.choices = [mock_choice]

        clean_text, emotion = mgr.chat_with_emotion("I got a new pet!")
        self.assertEqual(emotion, "HAPPY")
        self.assertEqual(clean_text, "That sounds wonderful! I am so glad.")

    def test_chat_with_emotion_interactive(self) -> None:
        mgr = GroqChatManager()
        mock_choice = MagicMock()
        mock_choice.message.content = "[MEAL_BURGER] Yummy! Let's enjoy this delicious burger!"
        self.mock_groq_instance.chat.completions.create.return_value.choices = [mock_choice]

        clean_text, emotion = mgr.chat_with_emotion("Let's eat burgers!")
        self.assertEqual(emotion, "MEAL_BURGER")
        self.assertEqual(clean_text, "Yummy! Let's enjoy this delicious burger!")

    def test_chat_with_emotion_fallback(self) -> None:
        mgr = GroqChatManager()
        mock_choice = MagicMock()
        mock_choice.message.content = "Just a regular untagged response."
        self.mock_groq_instance.chat.completions.create.return_value.choices = [mock_choice]

        clean_text, emotion = mgr.chat_with_emotion("How is the weather?")
        self.assertEqual(emotion, "IDLE_NEUTRAL")
        self.assertEqual(clean_text, "Just a regular untagged response.")

    def test_chat_with_emotion_ignore(self) -> None:
        mgr = GroqChatManager()
        mock_choice = MagicMock()
        mock_choice.message.content = "[IGNORE]"
        self.mock_groq_instance.chat.completions.create.return_value.choices = [mock_choice]

        clean_text, emotion = mgr.chat_with_emotion("mummy khana de do")
        self.assertEqual(emotion, "IGNORE")
        self.assertEqual(clean_text, "")


if __name__ == "__main__":
    unittest.main()

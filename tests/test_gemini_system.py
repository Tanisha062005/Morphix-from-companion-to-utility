"""
Morphix Desktop Companion Robot - Gemini System Test Suite
==========================================================
File: tests/test_gemini_system.py
Description:
    Verifies NLU intent extraction, chat, JSON validation, retry logic,
    timeout handling, confidence validation, lazy loading AI registry,
    and routing execution logic for Morphix companion tasks.
"""

import os
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.ai.ai_registry import AIRegistry
from src.gemini.chat_memory import ChatMemory
from src.gemini.gemini_client import GeminiClient
from src.gemini.gemini_config import GeminiConfig
from src.gemini.gemini_intent_engine import GeminiIntentEngine
from src.gemini.intent_router import IntentRouter
from src.gemini.json_parser import JsonParser
from src.gemini.response_validator import ResponseValidator
from src.reminder.reminder_manager import ReminderManager
from src.todo.todo_manager import TodoManager


class TestGeminiSystem(unittest.TestCase):
    """
    Unit and mock-based integration test suite for Gemini AI Brain.
    """

    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.todo_json = Path(self.temp_dir.name) / "todo.json"
        self.rem_json = Path(self.temp_dir.name) / "reminders.json"

        self.todo_mgr = TodoManager(file_path=str(self.todo_json))
        self.reminder_mgr = ReminderManager(file_path=str(self.rem_json))

        # Patch GenAI client construction
        self.client_patcher = patch("google.genai.Client")
        self.mock_client_cls = self.client_patcher.start()
        self.mock_client = MagicMock()
        self.mock_client_cls.return_value = self.mock_client

        # Configure environment variables
        os.environ["GEMINI_API_KEY"] = "test_mock_api_key"
        os.environ["GEMINI_MODEL"] = "gemini-2.5-flash"

        self.client = GeminiClient()
        self.memory = ChatMemory()
        self.engine = GeminiIntentEngine(client=self.client, memory=self.memory)
        self.router = IntentRouter(
            todo_mgr=self.todo_mgr,
            reminder_mgr=self.reminder_mgr,
            gemini_engine=self.engine
        )

    def tearDown(self) -> None:
        self.client_patcher.stop()
        self.temp_dir.cleanup()
        if "GEMINI_API_KEY" in os.environ:
            del os.environ["GEMINI_API_KEY"]

    def test_api_initialization(self) -> None:
        self.assertTrue(self.client.is_available())
        self.assertEqual(self.client.model_name, GeminiConfig.DEFAULT_MODEL_NAME)

    def test_missing_api_key(self) -> None:
        # Temporarily clear key
        if "GEMINI_API_KEY" in os.environ:
            del os.environ["GEMINI_API_KEY"]

        disabled_client = GeminiClient()
        self.assertFalse(disabled_client.is_available())
        self.assertIsNone(disabled_client.client)

    def test_json_parser(self) -> None:
        raw_md = "```json\n{\"intent\": \"SHOW_TODO\", \"confidence\": 0.95}\n```"
        parsed = JsonParser.parse_gemini_json(raw_md)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["intent"], "SHOW_TODO")

        # Invalid JSON
        self.assertIsNone(JsonParser.parse_gemini_json("{invalid_json}"))

    def test_response_validator_confidence_check(self) -> None:
        # Confidence too low
        low_conf = {"intent": "SHOW_TODO", "confidence": 0.5}
        self.assertFalse(ResponseValidator.validate_intent_json(low_conf))

        # Confidence OK
        high_conf = {"intent": "SHOW_TODO", "confidence": 0.95}
        self.assertTrue(ResponseValidator.validate_intent_json(high_conf))

        # Missing confidence
        missing_conf = {"intent": "SHOW_TODO"}
        self.assertFalse(ResponseValidator.validate_intent_json(missing_conf))

    def test_response_validator_schemas(self) -> None:
        valid_add = {"intent": "ADD_TODO", "confidence": 0.9, "task": "finish robotics report"}
        self.assertTrue(ResponseValidator.validate_intent_json(valid_add))

        # Missing 'task' parameter
        invalid_add = {"intent": "ADD_TODO", "confidence": 0.9}
        self.assertFalse(ResponseValidator.validate_intent_json(invalid_add))

        valid_rem = {"intent": "SET_REMINDER", "confidence": 0.95, "message": "take medicine", "delay_seconds": 1800}
        self.assertTrue(ResponseValidator.validate_intent_json(valid_rem))

    def test_intent_extraction_successful(self) -> None:
        mock_response = MagicMock()
        self.mock_client.models.generate_content.return_value = mock_response
        mock_response.text = '{"intent": "ADD_TODO", "confidence": 0.98, "task": "finish OpenCV project"}'

        result = self.engine.process_transcript("add finish OpenCV project")
        self.assertEqual(result["intent"], "ADD_TODO")
        self.assertEqual(result["task"], "finish OpenCV project")

    def test_intent_extraction_retry_recovery(self) -> None:
        mock_response = MagicMock()
        self.mock_client.models.generate_content.side_effect = [
            Exception("Connection error"), # First attempt fails
            mock_response                  # Retry succeeds
        ]
        mock_response.text = '{"intent": "SHOW_TODO", "confidence": 0.95}'

        result = self.engine.process_transcript("show my todos")
        self.assertEqual(result["intent"], "SHOW_TODO")

    def test_intent_extraction_retry_max_reached(self) -> None:
        self.mock_client.models.generate_content.side_effect = [
            Exception("Error 1"),
            Exception("Error 2")
        ]
        result = self.engine.process_transcript("show my todos")
        self.assertEqual(result["intent"], "UNKNOWN")

    def test_chat_history_limit(self) -> None:
        mock_response = MagicMock()
        self.mock_client.models.generate_content.return_value = mock_response
        mock_response.text = "Hello!"

        # Insert 6 exchanges (12 messages)
        for i in range(6):
            self.engine.generate_chat_response(f"message {i}")

        history = self.memory.get_history()
        # Should be pruned to exactly 10 messages
        self.assertEqual(len(history), 10)

    def test_intent_router_todo_execution(self) -> None:
        intent = {"intent": "ADD_TODO", "confidence": 0.9, "task": "water the plants"}
        res = self.router.route(intent)
        self.assertEqual(res, "Todo added.")
        self.assertEqual(len(self.todo_mgr.get_pending_tasks()), 1)

    def test_intent_router_reminder_execution(self) -> None:
        intent = {"intent": "SET_REMINDER", "confidence": 0.95, "message": "renew passport", "delay_seconds": 3600}
        res = self.router.route(intent)
        self.assertEqual(res, "Reminder saved.")
        self.assertEqual(len(self.reminder_mgr.get_pending_reminders()), 1)

    def test_duplicate_transcript_deduplication(self) -> None:
        mock_response = MagicMock()
        self.mock_client.models.generate_content.return_value = mock_response
        mock_response.text = '{"intent": "SHOW_TODO", "confidence": 0.95}'

        # First request should call API
        r1 = self.engine.process_transcript("show todo")
        self.assertEqual(r1["intent"], "SHOW_TODO")
        self.assertEqual(self.mock_client.models.generate_content.call_count, 1)

        # Second request (within window) should bypass API
        r2 = self.engine.process_transcript("show todo")
        self.assertEqual(r2["intent"], "SHOW_TODO")
        self.assertEqual(self.mock_client.models.generate_content.call_count, 1)

    def test_ai_registry_lazy_initialization(self) -> None:
        registry = AIRegistry()
        factory_called = [False]

        def dummy_factory():
            factory_called[0] = True
            return "DummyAIInstance"

        registry.register_provider("Dummy", dummy_factory)
        self.assertFalse(factory_called[0])

        # Triggers instantiation on retrieve
        instance = registry.get_provider("Dummy")
        self.assertTrue(factory_called[0])
        self.assertEqual(instance, "DummyAIInstance")

    def test_dynamic_intent_registration(self) -> None:
        registry = AIRegistry()
        unregistered_data = {"intent": "WEATHER", "confidence": 0.95}
        self.assertFalse(ResponseValidator.validate_intent_json(unregistered_data))
        
        registry.register_intent("WEATHER")
        self.assertTrue(ResponseValidator.validate_intent_json(unregistered_data))

    def test_first_request_validation_success(self) -> None:
        client = GeminiClient()
        client.client = MagicMock()
        client.client.models.list.return_value = [MagicMock()]
        
        self.assertFalse(getattr(client, "_checked_validation", False))
        self.assertTrue(client.validate_key_if_first_time())
        self.assertTrue(getattr(client, "_checked_validation", True))
        self.assertTrue(client.is_available())

    def test_first_request_validation_failure(self) -> None:
        client = GeminiClient()
        client.client = MagicMock()
        client.client.models.list.side_effect = Exception("API Key Invalid")
        
        self.assertFalse(client.validate_key_if_first_time())
        self.assertFalse(client.is_available())

    def test_timeout_http_options(self) -> None:
        with patch("google.genai.Client") as mock_genai:
            os.environ["GEMINI_API_KEY"] = "test_mock_key_longer_than_15"
            GeminiConfig.TIMEOUT_SECONDS = 5.0
            client = GeminiClient()
            self.assertTrue(mock_genai.called)
            args, kwargs = mock_genai.call_args
            self.assertIn("http_options", kwargs)
            self.assertEqual(kwargs["http_options"].timeout, 5000)

    def test_validate_api_key(self) -> None:
        from src.gemini.gemini_client import validate_api_key
        
        # Test empty/None key
        self.assertFalse(validate_api_key(None)[0])
        self.assertFalse(validate_api_key("")[0])
        self.assertFalse(validate_api_key("   ")[0])
        
        # Test placeholders
        self.assertFalse(validate_api_key("your_api_key_here")[0])
        self.assertFalse(validate_api_key("INSERT_KEY_HERE")[0])
        self.assertFalse(validate_api_key("placeholder_value")[0])
        
        # Test length threshold
        self.assertFalse(validate_api_key("AIzaSyTooShort")[0])
        
        # Test incorrect prefixes
        self.assertFalse(validate_api_key("invalid_prefix_key_longer_than_15")[0])
        
        # Test valid keys
        self.assertTrue(validate_api_key("AIzaSyValidKeyLongerThan15")[0])
        self.assertTrue(validate_api_key("test_mock_api_key_longer_than_15")[0])

    def test_router_is_available(self) -> None:
        from src.gemini.intent_router import IntentRouter
        
        mock_engine = MagicMock()
        mock_client = MagicMock()
        mock_engine.client = mock_client
        
        router = IntentRouter(todo_mgr=MagicMock(), reminder_mgr=MagicMock(), gemini_engine=mock_engine)
        
        mock_client.is_available.return_value = True
        self.assertTrue(router.is_available())
        
        mock_client.is_available.return_value = False
        self.assertFalse(router.is_available())


if __name__ == "__main__":
    unittest.main()

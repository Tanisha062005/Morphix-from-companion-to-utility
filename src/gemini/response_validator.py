"""
Morphix Desktop Companion Robot - Gemini Response Validator
===========================================================
File: src/gemini/response_validator.py
Description:
    Enforces strict structural, data type, key schema, and confidence scoring
    checks on structured JSON intents returned by the Gemini Brain.
"""

from typing import Any, Dict
from src.gemini.gemini_config import GeminiConfig
from src.ai.ai_registry import AIRegistry


class ResponseValidator:
    """
    Validates dictionary shapes and intent parameters extracted by Gemini.
    """

    @staticmethod
    def validate_intent_json(data: Dict[str, Any]) -> bool:
        """
        Validates keys, parameter schemas, confidence values, and types.

        :param data: Decoded JSON dictionary.
        :return: True if valid, False otherwise.
        """
        if not isinstance(data, dict):
            return False

        if "intent" not in data or "confidence" not in data:
            return False

        intent = data.get("intent")
        confidence = data.get("confidence")

        if not isinstance(intent, str):
            return False
        if not isinstance(confidence, (int, float)):
            return False

        if confidence < GeminiConfig.CONFIDENCE_THRESHOLD:
            return False

        valid_intents = AIRegistry().get_supported_intents()
        if intent not in valid_intents:
            return False

        # Enforce schemas per intent type
        schemas = {
            "ADD_TODO": {
                "required": {"intent", "confidence", "task"},
                "allowed": {"intent", "confidence", "task"},
                "types": {"task": str}
            },
            "DELETE_TODO": {
                "required": {"intent", "confidence", "task"},
                "allowed": {"intent", "confidence", "task"},
                "types": {"task": str}
            },
            "EDIT_TODO": {
                "required": {"intent", "confidence", "old_item", "new_item"},
                "allowed": {"intent", "confidence", "old_item", "new_item"},
                "types": {"old_item": str, "new_item": str}
            },
            "COMPLETE_TODO": {
                "required": {"intent", "confidence", "task"},
                "allowed": {"intent", "confidence", "task"},
                "types": {"task": str}
            },
            "SHOW_TODO": {
                "required": {"intent", "confidence"},
                "allowed": {"intent", "confidence"},
                "types": {}
            },
            "SET_REMINDER": {
                "required": {"intent", "confidence", "message", "delay_seconds"},
                "allowed": {"intent", "confidence", "message", "delay_seconds"},
                "types": {"message": str, "delay_seconds": (int, float)}
            },
            "DELETE_REMINDER": {
                "required": {"intent", "confidence", "target"},
                "allowed": {"intent", "confidence", "target"},
                "types": {"target": str}
            },
            "SHOW_REMINDERS": {
                "required": {"intent", "confidence"},
                "allowed": {"intent", "confidence"},
                "types": {}
            },
            "CHAT": {
                "required": {"intent", "confidence", "query"},
                "allowed": {"intent", "confidence", "query"},
                "types": {"query": str}
            },
            "UNKNOWN": {
                "required": {"intent", "confidence"},
                "allowed": {"intent", "confidence"},
                "types": {}
            }
        }

        if intent not in schemas:
            # For dynamically registered intents not predefined in schemas,
            # we require at least 'intent' (str) and 'confidence' (int/float)
            return True

        schema = schemas[intent]
        data_keys = set(data.keys())

        # Check required/missing keys
        if not schema["required"].issubset(data_keys):
            return False

        # Check unexpected keys
        if not data_keys.issubset(schema["allowed"]):
            return False

        # Check field types and required parameters constraints
        for field, expected_type in schema["types"].items():
            val = data.get(field)
            if not isinstance(val, expected_type):
                return False
            # If expected string, also check it is not empty/whitespace
            if expected_type is str and not val.strip():
                return False
            # If delay_seconds, check it's positive
            if field == "delay_seconds" and val <= 0:
                return False

        return True

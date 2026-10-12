"""
Morphix Desktop Companion Robot - Facial Expression Manager
===========================================================
File: src/emotions/expression_manager.py
Description:
    Manages facial expression data and selection logic for Morphix.
    This module strictly adheres to the Single Responsibility Principle (SRP)
    and acts purely as a logic and data repository for expressions without
    controlling display hardware or external rendering engines.
"""

import threading
from typing import Any, Dict, List, Optional


class ExpressionManager:
    """
    Manages Morphix's facial expression states, emojis, and descriptions.
    """

    DEFAULT_EXPRESSION: str = "Neutral"
    _instance: Optional["ExpressionManager"] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "ExpressionManager":
        """Singleton accessor for shared ExpressionManager instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self, initial_expression: str = DEFAULT_EXPRESSION) -> None:
        """
        Initializes the ExpressionManager with supported expression definitions.

        :param initial_expression: Starting expression name (default: "Neutral").
        """
        self._current_expression: str = self.DEFAULT_EXPRESSION
        # Internal dictionary of supported expressions and their properties
        self._expressions: Dict[str, Dict[str, str]] = {
            "Neutral": {
                "name": "Neutral",
                "emoji": "😐",
                "description": "Standard relaxed expression, standing by."
            },
            "Happy": {
                "name": "Happy",
                "emoji": "😊",
                "description": "Smiling eyes and cheerful face."
            },
            "Sad": {
                "name": "Sad",
                "emoji": "😢",
                "description": "Slightly drooping eyes and solemn look."
            },
            "Angry": {
                "name": "Angry",
                "emoji": "😠",
                "description": "Furrowed eyebrows and intense eyes."
            },
            "Thinking": {
                "name": "Thinking",
                "emoji": "🤔",
                "description": "Glancing upward, analyzing input."
            },
            "Sleeping": {
                "name": "Sleeping",
                "emoji": "😴",
                "description": "Closed eyes, low activity state."
            },
            "Listening": {
                "name": "Listening",
                "emoji": "👂",
                "description": "Wide attentive eyes listening for user voice."
            },
            "Excited": {
                "name": "Excited",
                "emoji": "🤩",
                "description": "Sparkling wide eyes and vibrant reaction."
            },
            "Confused": {
                "name": "Confused",
                "emoji": "😕",
                "description": "Asymmetrical eyebrows and tilted gaze."
            },
            "Protective": {
                "name": "Protective",
                "emoji": "🛡️",
                "description": "Focused watchful eyes monitoring desk environment."
            }
        }

        # Set initial expression (fall back to Neutral if invalid)
        self.set_expression(initial_expression)

    def set_expression(self, expression_name: str) -> str:
        """
        Sets the active facial expression. Automatically falls back to "Neutral"
        if an invalid or unknown expression name is provided.

        :param expression_name: Name of expression to switch to (case-insensitive).
        :return: The active expression name after updating.
        """
        formatted_name = expression_name.strip().capitalize() if expression_name else ""

        if formatted_name in self._expressions:
            self._current_expression = formatted_name
        else:
            self._current_expression = self.DEFAULT_EXPRESSION

        return self._current_expression

    def get_current_expression(self) -> str:
        """
        Returns the name of Morphix's current active facial expression.

        :return: String name of current expression (e.g., "Happy").
        """
        return self._current_expression

    def get_emoji(self) -> str:
        """
        Returns the emoji character for the current active expression.

        :return: Emoji character string.
        """
        return self._expressions[self._current_expression]["emoji"]

    def get_description(self) -> str:
        """
        Returns the short textual description of the current active expression.

        :return: Description string.
        """
        return self._expressions[self._current_expression]["description"]

    def reset_expression(self) -> None:
        """
        Resets the active facial expression back to the default state ("Neutral").
        """
        self._current_expression = self.DEFAULT_EXPRESSION

    def get_available_expressions(self) -> List[str]:
        """
        Returns a list of all supported facial expression names.

        :return: List of expression name strings.
        """
        return list(self._expressions.keys())

    def to_eye_state(self) -> Any:
        """
        Maps the current expression to the corresponding EyeState enum value for EyeAnimator.
        """
        try:
            from src.display.eye_animator import EyeState
            mapping = {
                "Neutral": EyeState.IDLE_NEUTRAL,
                "Happy": EyeState.HAPPY,
                "Sad": EyeState.SICK_COLD,
                "Angry": EyeState.ANGRY,
                "Thinking": EyeState.THINKING,
                "Sleeping": EyeState.DEEP_SLEEP,
                "Listening": EyeState.LISTENING,
                "Excited": EyeState.HEART_EYES,
                "Confused": EyeState.CONFUSED,
                "Protective": EyeState.SENTRY_SCAN,
            }
            return mapping.get(self._current_expression, EyeState.IDLE_NEUTRAL)
        except Exception:
            return None


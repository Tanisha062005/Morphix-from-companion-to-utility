"""
Morphix Desktop Companion Robot - Emotion Engine
=================================================
File: src/ai/personality/emotion.py
Description:
    Manages Morphix's emotional states, emojis, and descriptions.
    This module strictly adheres to the Single Responsibility Principle (SRP)
    and serves as the single source of truth for robot emotional state management.
"""

from typing import Dict


class EmotionEngine:
    """
    Manages the current emotional state of Morphix.
    Provides methods to set, retrieve, and inspect active emotions.
    """

    DEFAULT_EMOTION: str = "Happy"

    def __init__(self, default_emotion: str = DEFAULT_EMOTION) -> None:
        """
        Initializes the EmotionEngine with all supported emotions and sets the initial state.

        :param default_emotion: The baseline emotion for the robot (default: "Happy").
        """
        # Internal dictionary storing supported emotions and their attributes
        self._emotions: Dict[str, Dict[str, str]] = {
            "Happy": {
                "name": "Happy",
                "emoji": "😊",
                "description": "Feeling cheerful, content, and ready to interact."
            },
            "Sad": {
                "name": "Sad",
                "emoji": "😢",
                "description": "Feeling down, low energy, or needing encouragement."
            },
            "Excited": {
                "name": "Excited",
                "emoji": "🤩",
                "description": "Energetic, enthusiastic, and ready for action!"
            },
            "Thinking": {
                "name": "Thinking",
                "emoji": "🤔",
                "description": "Processing information, solving a problem, or searching."
            },
            "Listening": {
                "name": "Listening",
                "emoji": "👂",
                "description": "Attentively listening for voice commands or user input."
            },
            "Sleeping": {
                "name": "Sleeping",
                "emoji": "😴",
                "description": "In low-power standby mode, resting eyes."
            },
            "Confused": {
                "name": "Confused",
                "emoji": "😕",
                "description": "Puzzled or unable to understand an input or command."
            },
            "Protective": {
                "name": "Protective",
                "emoji": "🛡️",
                "description": "On alert, monitoring desk environment and surrounding space."
            }
        }

        # Set initial default emotion
        self._default_emotion: str = (
            default_emotion if default_emotion in self._emotions else self.DEFAULT_EMOTION
        )
        self._current_emotion: str = self._default_emotion

    def set_emotion(self, emotion_name: str) -> bool:
        """
        Sets Morphix's active emotion.

        :param emotion_name: Name of the emotion to switch to (e.g., "Excited").
        :return: True if successfully updated, False if the emotion is invalid.
        """
        formatted_name = emotion_name.strip().capitalize()

        if formatted_name in self._emotions:
            self._current_emotion = formatted_name
            return True
        return False

    def get_current_emotion(self) -> str:
        """
        Returns the name of Morphix's current active emotion.

        :return: String name of the current emotion (e.g., "Happy").
        """
        return self._current_emotion

    def get_emoji(self) -> str:
        """
        Returns the emoji representing the current active emotion.

        :return: Emoji character string (e.g., "😊").
        """
        return self._emotions[self._current_emotion]["emoji"]

    def get_description(self) -> str:
        """
        Returns a short textual description of the current active emotion.

        :return: Description string.
        """
        return self._emotions[self._current_emotion]["description"]

    def reset_to_default(self) -> None:
        """
        Resets Morphix's current emotion back to the default state.
        """
        self._current_emotion = self._default_emotion

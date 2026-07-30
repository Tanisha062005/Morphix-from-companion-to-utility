"""
Morphix Desktop Companion Robot - Personality Module
=====================================================
File: personality.py
Description:
    Manages the core identity, traits, mood state, and conversational 
    message templates for the Morphix desktop companion robot.
"""

import random
from typing import List, Optional


class Personality:
    """
    Manages Morphix's personality traits, emotional moods, and dialogue responses.
    """

    def __init__(
        self,
        name: str = "Morphix",
        traits: Optional[List[str]] = None,
        speaking_style: str = "Friendly, curious, and playful",
        initial_mood: str = "Happy"
    ) -> None:
        """
        Initializes Morphix's personality attributes and message categories.

        :param name: Name of the robot (default: "Morphix").
        :param traits: List of personality traits.
        :param speaking_style: Description of how the robot speaks.
        :param initial_mood: Starting mood state (default: "Happy").
        """
        # Core identity attributes
        self.name: str = name
        self.traits: List[str] = traits if traits is not None else [
            "Helpful",
            "Curious",
            "Empathetic",
            "Playful",
            "Observant"
        ]
        self.speaking_style: str = speaking_style

        # Mood tracking
        self.available_moods: List[str] = [
            "Happy",
            "Curious",
            "Excited",
            "Thoughtful",
            "Focused",
            "Sleepy",
            "Confused"
        ]
        self.current_mood: str = initial_mood if initial_mood in self.available_moods else "Happy"

        # Pre-configured message pools
        self.greetings: List[str] = [
            f"Hello! I'm {self.name}. How can I help you today?",
            f"Beep boop! Good to see you!",
            f"Hi there! {self.name} is online and ready!",
            f"Greetings! Hope you are having a great day!"
        ]

        self.goodbyes: List[str] = [
            f"Goodbye for now! {self.name} will be right here when you return.",
            "Powering down to standby mode. See you soon!",
            "Take care! Have an awesome day!",
            "Catch you later!"
        ]

        self.idle_messages: List[str] = [
            "Just looking around the room...",
            "Wondering what project we'll work on next!",
            "Blinking eyes... still here whenever you need me.",
            "Standing by and listening..."
        ]

        self.error_messages: List[str] = [
            "Oops! Something went wrong on my end. Let me try again.",
            "Beep! I encountered a small glitch. Could you repeat that?",
            "Hmm, I got a bit confused for a second.",
            "My bad! Something didn't work right."
        ]

        self.small_talk_messages: List[str] = [
            "Did you know? Building robots is one of my favorite topics!",
            "I enjoy helping out on your desk.",
            "Every day is a good day to learn something new!",
            "I'm feeling pretty good today! How about you?"
        ]

    def introduce(self) -> str:
        """Returns a formatted introductory string about Morphix."""
        traits_str = ", ".join(self.traits)
        return (
            f"Hello! I am {self.name}.\n"
            f"Traits: {traits_str}.\n"
            f"Speaking Style: {self.speaking_style}.\n"
            f"Current Mood: {self.current_mood}."
        )

    def greet(self) -> str:
        """Returns a random greeting message."""
        return random.choice(self.greetings)

    def goodbye(self) -> str:
        """Returns a random goodbye message."""
        return random.choice(self.goodbyes)

    def idle_message(self) -> str:
        """Returns a random idle message."""
        return random.choice(self.idle_messages)

    def error_message(self) -> str:
        """Returns a random error response message."""
        return random.choice(self.error_messages)

    def small_talk(self) -> str:
        """Returns a random small-talk response message."""
        return random.choice(self.small_talk_messages)

    def set_mood(self, new_mood: str) -> bool:
        """
        Updates Morphix's current mood if the given mood is valid.

        :param new_mood: Mood string to set.
        :return: True if updated successfully, False otherwise.
        """
        if new_mood in self.available_moods:
            self.current_mood = new_mood
            return True
        return False

    def get_current_mood(self) -> str:
        """Returns Morphix's current active mood."""
        return self.current_mood

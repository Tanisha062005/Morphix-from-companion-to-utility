"""
Morphix Desktop Companion Robot - Wake Word Detector Test Suite
================================================================
File: tests/test_wake_word.py
Description:
    Unit test suite for WakeWordDetector.
    Verifies text normalization, exact wake word triggers, near match triggers,
    false positive rejections, and custom similarity threshold configuration.
"""

import logging
from pathlib import Path
import sys
import unittest

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import Config
from src.voice.wake_word import WakeWordDetector

# Setup logger for test runner
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [TestWakeWord]: %(message)s"
)
logger = logging.getLogger("TestWakeWord")


class TestWakeWordDetector(unittest.TestCase):
    """
    Test cases for WakeWordDetector matching accuracy and configuration.
    """

    def setUp(self) -> None:
        self.detector = WakeWordDetector(
            wake_word=Config.WAKE_PHRASE,
            threshold=Config.SIMILARITY_THRESHOLD,
            aliases=Config.ACCEPTED_ALIASES
        )

    def test_text_normalization(self) -> None:
        """Verifies text normalization converts to lowercase, strips punctuation, and collapses whitespace."""
        self.assertEqual(WakeWordDetector.normalize_text("  Hey, Morphix!  "), "hey morphix")
        self.assertEqual(WakeWordDetector.normalize_text("HE... MIGHT... FIX???"), "he might fix")
        self.assertEqual(WakeWordDetector.normalize_text(""), "")
        self.assertEqual(WakeWordDetector.normalize_text(None), "")

    def test_exact_wake_word_detection(self) -> None:
        """Verifies exact matches trigger wake detection with score 100.0."""
        score, matched_phrase, matched_alias, similarity, is_detected = self.detector.compute_similarity("Hey Morphix")
        self.assertEqual(score, 100.0)
        self.assertTrue(self.detector.detect("Hey Morphix"))

    def test_near_matches_and_accepted_aliases(self) -> None:
        """
        Verifies near match pronunciations and accepted aliases trigger wake detection.
        Target examples from user prompt:
            - Hey Morphix
            - Hey Morfix
            - Hey Morphics
            - Hey More Fix
            - Hey More Pics
            - He Might Fix
            - He More Fix
            - he morphics
            - he more
            - he motivates
        """
        near_matches = [
            "Hey Morphix",
            "Hey Morfix",
            "Hey Morphics",
            "Hey More Fix",
            "Hey More Pics",
            "He Might Fix",
            "He More Fix",
            "he morphics",
            "he more",
            "he motivates",
            "he wont fix",
            "hey morefiks"
        ]

        for phrase in near_matches:
            is_wake = self.detector.detect(phrase)
            self.assertTrue(
                is_wake,
                f"Phrase '{phrase}' should trigger wake detection, but returned NO."
            )

    def test_false_positives_should_not_trigger(self) -> None:
        """
        Verifies unrelated non-wake phrases DO NOT trigger wake detection.
        Target examples from user prompt:
            - Hello
            - Good Morning
            - What's the weather
            - Open Chrome
            - Play Music
        """
        false_positives = [
            "Hello",
            "Good Morning",
            "What's the weather",
            "Open Chrome",
            "Play Music",
            "Turn on the lights",
            "How are you today",
            "Stop playing music"
        ]

        for phrase in false_positives:
            is_wake = self.detector.detect(phrase)
            self.assertFalse(
                is_wake,
                f"False positive phrase '{phrase}' should NOT trigger wake detection."
            )

    def test_custom_threshold_and_aliases(self) -> None:
        """Verifies custom threshold configuration and custom alias lists work as expected."""
        strict_detector = WakeWordDetector(
            wake_word="hey morphix",
            threshold=0.99,
            aliases=["hey morphix"]
        )

        self.assertTrue(strict_detector.detect("hey morphix"))
        self.assertFalse(strict_detector.detect("he might fix"))


def run_tests() -> None:
    unittest.main(exit=False)


if __name__ == "__main__":
    run_tests()

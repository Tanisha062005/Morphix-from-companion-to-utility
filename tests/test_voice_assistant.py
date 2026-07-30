"""
Morphix Desktop Companion Robot - VoiceAssistant Test Suite
============================================================
File: tests/test_voice_assistant.py
Description:
    Unit and integration tests for the VoiceAssistant coordinator class,
    verifying wake word detection, sound player handling, Piper TTS greetings,
    and STT listening state transitions.
"""

import logging
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock

# Ensure project root is in sys.path for clean import resolution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import Config
from src.audio.sound_player import SoundPlayer
from src.voice.speech_to_text import SpeechToText
from src.voice.text_to_speech import TextToSpeech
from src.voice.voice_assistant import VoiceAssistant
from src.voice.wake_word import WakeWordDetector

# Configure test runner logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [TestVoiceAssistant]: %(message)s"
)
logger = logging.getLogger("TestVoiceAssistant")


class TestVoiceAssistant(unittest.TestCase):
    """
    Test suite for VoiceAssistant coordination, missing file resilience,
    greeting triggers, and STT state transitions.
    """

    def setUp(self) -> None:
        """
        Sets up clean mock objects and dependencies for each test case.
        """
        self.mock_detector = WakeWordDetector()
        self.mock_sound_player = MagicMock(spec=SoundPlayer)
        self.mock_tts = MagicMock(spec=TextToSpeech)
        self.mock_stt = SpeechToText()

        # Instantiate assistant under test with injected mocks
        self.assistant = VoiceAssistant(
            detector=self.mock_detector,
            sound_player=self.mock_sound_player,
            tts=self.mock_tts,
            stt=self.mock_stt
        )

    def test_sounds_directory_auto_created(self) -> None:
        """
        Verifies that assets/sounds/ directory is automatically created.
        """
        sound_player = SoundPlayer()
        sounds_dir = sound_player.get_sounds_dir()
        self.assertTrue(sounds_dir.exists(), "assets/sounds/ directory should exist.")

    def test_wake_word_detected_flow_with_missing_sound(self) -> None:
        """
        Verifies flow execution when wake word is detected and morphix_voice.wav is missing.
        Should handle missing sound cleanly, trigger TTS greeting, and start STT listening.
        """
        # Configure sound player to return False for missing file
        self.mock_sound_player.play_sound.return_value = False

        # Execute assistant workflow with wake word text
        result = self.assistant.process_wake_word("Hey Morphix")

        self.assertTrue(result, "process_wake_word should return True when wake word is detected.")
        self.mock_sound_player.play_sound.assert_called_once()
        self.mock_tts.speak.assert_called_once_with(Config.WAKE_RESPONSE)
        self.assertTrue(self.mock_stt.is_listening(), "STT should be set to listening mode after greeting.")

    def test_wake_word_detected_flow_with_existing_sound(self) -> None:
        """
        Verifies flow execution when morphix_voice.wav exists and plays successfully.
        """
        # Configure sound player to simulate successful sound playback
        self.mock_sound_player.play_sound.return_value = True

        result = self.assistant.process_wake_word("Hey Morphix, how are you?")

        self.assertTrue(result, "process_wake_word should return True.")
        self.mock_sound_player.play_sound.assert_called_once_with(self.assistant.wake_sound_path)
        self.mock_tts.speak.assert_called_once_with(Config.WAKE_RESPONSE)
        self.assertTrue(self.mock_stt.is_listening(), "STT listening should be active.")


    def test_non_wake_word_input(self) -> None:
        """
        Verifies that non-wake-word input does not trigger bark playback, TTS, or STT state change.
        """
        result = self.assistant.process_wake_word("What is the current time?")

        self.assertFalse(result, "process_wake_word should return False when wake word is absent.")
        self.mock_sound_player.play_sound.assert_not_called()
        self.mock_tts.speak.assert_not_called()
        self.assertFalse(self.mock_stt.is_listening(), "STT listening should remain inactive.")

    def test_text_normalization(self) -> None:
        """
        Verifies text normalization: lowercase, punctuation removal, and space trimming.
        """
        normalized = WakeWordDetector.normalize_text("  HEY,  Morphix!!!  ")
        self.assertEqual(normalized, "hey morphix")

    def test_fuzzy_wake_word_matching(self) -> None:
        """
        Verifies multi-algorithm RapidFuzz similarity matching:
        Target examples that MUST trigger (Final Score >= 80.0):
        - hey morphix
        - he might fix
        - he wont fix
        - hey more fix
        - hey morfix
        - hey morphics
        - hey morefiks

        Target examples that MUST NOT trigger:
        - hello
        - market
        - alexa
        - computer
        - nobody
        - good morning
        - what
        - pokemon
        - motivate
        """
        detector = WakeWordDetector(threshold=80.0)

        must_trigger = [
            "hey morphix",
            "he might fix",
            "he wont fix",
            "hey more fix",
            "hey morfix",
            "hey morphics",
            "hey morefiks"
        ]
        for valid_str in must_trigger:
            with self.subTest(valid_str=valid_str):
                score, matched_phrase, matched_alias, similarity, is_detected = detector.compute_similarity(valid_str)
                self.assertGreaterEqual(score, 80.0, f"Input '{valid_str}' should score >= 80.0 (Got {score:.1f}).")
                self.assertTrue(detector.detect(valid_str), f"Input '{valid_str}' SHOULD trigger wake workflow.")

        must_not_trigger = [
            "hello",
            "market",
            "alexa",
            "computer",
            "nobody",
            "good morning",
            "what",
            "pokemon",
            "motivate"
        ]
        for invalid_str in must_not_trigger:
            with self.subTest(invalid_str=invalid_str):
                self.assertFalse(detector.detect(invalid_str), f"Input '{invalid_str}' MUST NOT trigger wake workflow.")



    def test_no_crash_on_empty_or_invalid_input(self) -> None:
        """
        Verifies that empty string, None, or non-string inputs are handled gracefully.
        """
        self.assertFalse(self.assistant.process_wake_word(""))

        self.assertFalse(self.assistant.process_wake_word(None))
        self.assertFalse(self.assistant.process_wake_word(12345))



def run_tests() -> None:
    """
    Executes TestVoiceAssistant test runner directly.
    """
    unittest.main(exit=False)


if __name__ == "__main__":
    run_tests()

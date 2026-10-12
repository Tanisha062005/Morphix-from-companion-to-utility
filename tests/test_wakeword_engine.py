"""
Morphix Desktop Companion Robot - WakeWordEngine Test Suite
=============================================================
File: tests/test_wakeword_engine.py
Description:
    Unit and integration tests for BaseWakeWordEngine, OpenWakeWordEngine,
    and WakeWordManager classes. Verifies missing model handling, swappable engine backends,
    and VoiceAssistant integration.
"""

import logging
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import Config
from src.audio.sound_player import SoundPlayer
from src.voice.speech_to_text import SpeechToText
from src.voice.text_to_speech import TextToSpeech
from src.voice.voice_assistant import VoiceAssistant
from src.wakeword.wakeword_engine import BaseWakeWordEngine, OpenWakeWordEngine
from src.wakeword.wakeword_manager import WakeWordManager

# Setup logger for test runner
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [TestWakeWordEngine]: %(message)s"
)
logger = logging.getLogger("TestWakeWordEngine")


class DummyCustomEngine(BaseWakeWordEngine):
    """
    Mock/Dummy custom wake word engine demonstrating backend swappability
    (e.g., Porcupine or Custom TensorFlow Lite).
    """

    def __init__(self, name: str = "DummyCustomEngine") -> None:
        self.name: str = name
        self._active: bool = False
        self._initialized: bool = False

    def initialize(self) -> bool:
        self._initialized = True
        return True

    def start(self) -> None:
        self._active = True

    def stop(self) -> None:
        self._active = False

    def is_detected(self, audio_frame: str = None) -> bool:
        return self._active and self._initialized and audio_frame == "trigger_wakeword"

    def get_model_path(self) -> Path:
        return Path("models/wakeword/dummy.tflite")

    def get_engine_name(self) -> str:
        return self.name


class TestWakeWordEngine(unittest.TestCase):
    """
    Test suite for modular WakeWordEngine architecture.
    """

    def test_openwakeword_engine_missing_model_behavior(self) -> None:
        """
        Verifies that OpenWakeWordEngine handles missing custom models gracefully.
        Should log missing model message and return False without crashing.
        """
        engine = OpenWakeWordEngine(model_dir="models/wakeword/non_existent_dir")
        success = engine.initialize()

        self.assertFalse(success, "initialize() should return False when custom model is missing.")
        self.assertIsNone(engine.get_model_path(), "get_model_path() should return None.")
        self.assertFalse(engine.is_detected(b"pcm_data"), "is_detected() should return False when model is missing.")

    def test_wakeword_manager_with_swappable_engine(self) -> None:
        """
        Verifies that WakeWordManager seamlessly works with swappable engine backends.
        """
        custom_engine = DummyCustomEngine("PorcupineEngine")
        manager = WakeWordManager(engine=custom_engine)

        self.assertTrue(manager.initialize())
        self.assertEqual(manager.get_engine_name(), "PorcupineEngine")

        manager.start()
        self.assertTrue(manager.is_listening())
        self.assertTrue(manager.is_detected("trigger_wakeword"))
        self.assertFalse(manager.is_detected("normal_speech"))

        manager.stop()
        self.assertFalse(manager.is_listening())
        self.assertFalse(manager.is_detected("trigger_wakeword"))

    def test_voice_assistant_integration_with_wakeword_manager(self) -> None:
        """
        Verifies VoiceAssistant interacts with WakeWordManager cleanly.
        """
        custom_engine = DummyCustomEngine("OpenWakeWordEngine")
        manager = WakeWordManager(engine=custom_engine)
        manager.initialize()
        manager.start()

        mock_sound_player = MagicMock(spec=SoundPlayer)
        mock_tts = MagicMock(spec=TextToSpeech)
        mock_stt = SpeechToText()

        assistant = VoiceAssistant(
            wake_word_manager=manager,
            sound_player=mock_sound_player,
            tts=mock_tts,
            stt=mock_stt
        )

        success = assistant.trigger_wake_sequence()

        self.assertTrue(success)
        mock_sound_player.play_sound.assert_called_once_with(assistant.wake_sound_path)
        mock_tts.speak.assert_called_once_with(Config.WAKE_RESPONSE)
    def test_fallback_wakeword_engine_integration(self) -> None:
        """
        Verifies FallbackWakeWordEngine initializes GoogleSpeechEngine and detects wake words cleanly.
        """
        from src.wakeword.wakeword_engine import FallbackWakeWordEngine
        engine = FallbackWakeWordEngine(target_phrase="Hey Morphix")
        self.assertTrue(engine.initialize())
        engine.start()

        # Mock STT transcribe to return "hey morphix"
        with unittest.mock.patch.object(engine._stt, "transcribe", return_value="hey morphix"):
            self.assertTrue(engine.is_detected(b"fake_audio_frame"))

        # Verify GoogleSpeechEngine has _transcribe_audio attribute
        self.assertTrue(hasattr(engine._stt.engine, "_transcribe_audio"))
        self.assertTrue(hasattr(engine._stt.engine, "transcribe"))


def run_tests() -> None:
    """
    Executes TestWakeWordEngine directly.
    """
    unittest.main(exit=False)


if __name__ == "__main__":
    run_tests()

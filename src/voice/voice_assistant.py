"""
Morphix Desktop Companion Robot - Voice Assistant Coordinator
===============================================================
File: src/voice/voice_assistant.py
Description:
    Coordinates the voice interaction workflow for Morphix without implementing
    low-level recognition or synthesis directly. Follows the Single Responsibility
    Principle by orchestrating WakeWordManager / WakeWordDetector, SoundPlayer,
    TextToSpeech (Piper), and SpeechToText (Vosk) modules.
"""

import logging
from pathlib import Path
import sys
from typing import Optional, Union

from config import Config
from src.audio.sound_player import SoundPlayer
from src.voice.speech_to_text import SpeechToText
from src.voice.text_to_speech import TextToSpeech
from src.voice.wake_word import WakeWordDetector
from src.wakeword.wakeword_manager import WakeWordManager

# Setup logger for VoiceAssistant module
logger = logging.getLogger("Morphix.VoiceAssistant")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [VoiceAssistant]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class VoiceAssistant:
    """
    Coordinator class for Morphix voice interaction workflows.
    Orchestrates wake word detection engines, audio sound playback, Piper TTS greeting,
    and Vosk STT listening state transitions.
    """

    def __init__(
        self,
        detector: Optional[WakeWordDetector] = None,
        wake_word_manager: Optional[WakeWordManager] = None,
        sound_player: Optional[SoundPlayer] = None,
        tts: Optional[TextToSpeech] = None,
        stt: Optional[SpeechToText] = None,
        wake_sound_path: Optional[Union[str, Path]] = None,
        bark_sound_path: Optional[Union[str, Path]] = None,
        greeting: Optional[str] = None
    ) -> None:
        """
        Initializes the VoiceAssistant coordinator with required dependencies via Dependency Injection.

        :param detector: Instance of WakeWordDetector (fallback compatibility).
        :param wake_word_manager: Instance of WakeWordManager for dedicated wake-word engines.
        :param sound_player: Instance of SoundPlayer (default: instantiated automatically).
        :param tts: Instance of TextToSpeech (default: instantiated automatically).
        :param stt: Instance of SpeechToText (default: instantiated automatically).
        :param wake_sound_path: Custom path to wake sound file (default: Config.WAKE_SOUND / assets/sounds/morphix_voice.wav).
        :param bark_sound_path: Alias for wake_sound_path for backward compatibility.
        :param greeting: Custom wake greeting string (default: Config.WAKE_RESPONSE).
        """
        self.wake_word_manager: WakeWordManager = wake_word_manager if wake_word_manager else WakeWordManager()
        self.detector: WakeWordDetector = detector if detector else WakeWordDetector()
        self.sound_player: SoundPlayer = sound_player if sound_player else SoundPlayer()
        self.tts: TextToSpeech = tts if tts else TextToSpeech()
        self.stt: SpeechToText = stt if stt else SpeechToText()

        # Target wake audio sound file path (loaded from Config.WAKE_SOUND by default)
        chosen_path = wake_sound_path or bark_sound_path
        if chosen_path:
            self.wake_sound_path: Path = Path(chosen_path).resolve()
        else:
            default_config_path = getattr(Config, "WAKE_SOUND", "assets/sounds/morphix_voice.wav")
            self.wake_sound_path: Path = Path(default_config_path).resolve()

        # Configurable greeting response string (defaults to Config.WAKE_RESPONSE)
        self.greeting: str = greeting if greeting else getattr(Config, "WAKE_RESPONSE", "Hello Tanisha! I'm listening.")

        logger.info("VoiceAssistant initialized cleanly. Wake sound path set to: %s", self.wake_sound_path)

    @property
    def bark_sound_path(self) -> Path:
        """Alias for wake_sound_path for backward compatibility."""
        return self.wake_sound_path

    def trigger_wake_sequence(self) -> bool:
        """
        Executes the wake workflow sequence after a wake word event occurs:
            1. Play assets/sounds/morphix_voice.wav completely (blocking playback).
            2. Speak greeting ("Hello Tanisha! I'm listening.") using Piper TTS.
            3. Start Vosk Speech-to-Text listening mode to process user's command.

        :return: True if sequence executed successfully.
        """
        logger.info("Triggering wake response sequence.")

        # 1. Play morphix_voice.wav sound (blocking, graceful fallback if missing)
        sound_played = self.sound_player.play_sound(self.wake_sound_path)
        if not sound_played:
            logger.info("Wake sound was skipped or completed without playback.")

        # 2. Speak greeting response using Piper TextToSpeech
        logger.info("Speaking wake response: '%s'", self.greeting)
        self.tts.speak(self.greeting)

        # 3. Enable Vosk Speech-to-Text listening state for next command
        self.stt.start_listening()
        logger.info("Vosk Speech-to-Text listening mode enabled (is_listening=%s).", self.stt.is_listening())

        return True

    def process_wake_word(self, text_input: str) -> bool:
        """
        Processes text input for wake word detection (backward compatibility & string evaluation).

        :param text_input: User speech transcript input to analyze for wake word.
        :return: True if wake word was detected and sequence executed, False otherwise.
        """
        if not text_input or not isinstance(text_input, str):
            return False

        is_detected = self.detector.detect(text_input)
        if not is_detected:
            return False

        logger.info("Wake word detected in text input: '%s'", text_input)
        return self.trigger_wake_sequence()

"""
Morphix Desktop Companion Robot - Speech-To-Text Coordinator
=============================================================
File: src/voice/speech_to_text.py
Description:
    Speech-To-Text orchestrator that wraps the active modular SpeechEngine backend
    (WhisperEngine) and handles transcribing files and bytes.
"""

import logging
import sys
from typing import Any, Optional, Union

from config import Config
from src.voice.speech_engine import SpeechEngine
from src.voice.whisper_engine import WhisperEngine

# Setup logger for SpeechToText module
logger = logging.getLogger("Morphix.STT")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [STT]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class SpeechToText:
    """
    Orchestration class for Speech-To-Text.
    Delegates to WhisperEngine (or another config-defined SpeechEngine).
    """

    def __init__(self, engine: Optional[SpeechEngine] = None) -> None:
        """
        Initializes SpeechToText with a configured engine.
        """
        self._listening: bool = False
        self._last_transcript: str = ""
        self.sample_rate: int = 16000
        
        self.engine: SpeechEngine = engine if engine else WhisperEngine()
        self.initialize()

    def initialize(self) -> bool:
        """
        Initializes the underlying speech engine backend.
        """
        return self.engine.initialize()

    def start_listening(self) -> None:
        """Enables listening state."""
        self._listening = True

    def stop_listening(self) -> None:
        """Disables listening state."""
        self._listening = False

    def is_listening(self) -> bool:
        """Returns whether the STT processor is active."""
        return self._listening

    def create_recognizer(self) -> Optional[Any]:
        """
        No-op method for Vosk compatibility.
        """
        return None

    def transcribe(self, audio_source: Union[str, bytes]) -> str:
        """
        Delegates the transcription of WAV file path or raw WAV bytes to the active SpeechEngine.
        """
        if not audio_source:
            return ""

        transcript = self.engine.transcribe(audio_source)
        self._last_transcript = transcript
        return transcript

    def get_last_transcript(self) -> str:
        """Retrieves the last transcribed text."""
        return self._last_transcript

    def clear_last_transcript(self) -> None:
        """Clears the last transcribed text."""
        self._last_transcript = ""

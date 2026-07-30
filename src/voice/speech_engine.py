"""
Morphix Desktop Companion Robot - Speech Engine Interface
===========================================================
File: src/voice/speech_engine.py
Description:
    Defines the SpeechEngine abstract base class, decoupling STT implementations
    (like Vosk or Whisper.cpp) from the rest of the application.
"""

from abc import ABC, abstractmethod
from typing import Union


class SpeechEngine(ABC):
    """
    Abstract Base Class defining the required interface for Speech-to-Text engines.
    """

    @abstractmethod
    def initialize(self) -> bool:
        """
        Initializes the speech engine and verifies required model files.

        :return: True if engine successfully initialized, False otherwise.
        """
        pass

    @abstractmethod
    def transcribe(self, audio_source: Union[str, bytes]) -> str:
        """
        Transcribes the provided audio source (file path or raw bytes) and returns 
        the complete finalized string.

        :param audio_source: WAV file path or raw WAV bytes.
        :return: Fully transcribed text string.
        """
        pass

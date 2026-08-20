"""
Morphix Desktop Companion Robot - Speech Engine Interface
===========================================================
File: src/voice/speech_engine.py
Description:
    Defines the SpeechEngine abstract base class, decoupling STT implementations
    from the rest of the application.
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
    def transcribe(self, audio_source: Union[str, bytes], language: str = "en-IN") -> str:
        """
        Transcribes the provided audio source (file path or raw bytes) and returns 
        the complete finalized string.

        :param audio_source: WAV file path or raw WAV bytes.
        :param language: BCP-47 language tag (e.g. 'en-IN', 'hi-IN').
        :return: Fully transcribed text string.
        """
        pass

"""
Morphix Desktop Companion Robot - Speech-To-Text Coordinator
=============================================================
File: src/voice/speech_to_text.py
Description:
    Speech-To-Text orchestrator backed exclusively by Google Web Speech API
    (speech_recognition library). No local model files required.

    Bilingual pipeline:
      1. Primary pass  → en-IN  (English / Latin-script Hinglish)
      2. Fallback pass → hi-IN  (triggered when confidence < 0.65 or primary fails)
"""

from __future__ import annotations

import io
import logging
import sys
from typing import Any, Optional, Union

import speech_recognition as sr

from src.voice.speech_engine import SpeechEngine

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------
logger = logging.getLogger("Morphix.STT")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] [STT]: %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


# ---------------------------------------------------------------------------
# Google Speech Engine — implements the SpeechEngine ABC
# ---------------------------------------------------------------------------

class GoogleSpeechEngine(SpeechEngine):
    """
    SpeechEngine implementation backed by Google Web Speech API via the
    `speech_recognition` library.

    Bilingual pipeline:
      • Pass 1: recognize_google(language="en-IN", show_all=True)
        - If result confidence >= 0.65 → return transcript
        - If confidence < 0.65 → fall through to hi-IN pass
      • Pass 2: recognize_google(language="hi-IN")
        - Used as fallback for Hindi / low-confidence audio
    """

    CONFIDENCE_THRESHOLD: float = 0.65

    def __init__(self) -> None:
        self._recognizer = sr.Recognizer()
        # Tweak default energy/pause thresholds for typical desktop microphones
        self._recognizer.energy_threshold = 300
        self._recognizer.dynamic_energy_threshold = True
        self._recognizer.pause_threshold = 0.8

    # ------------------------------------------------------------------
    # SpeechEngine ABC implementation
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """
        No model files to load — Google STT is network-based.
        Always returns True immediately.
        """
        logger.info("GoogleSpeechEngine ready (network STT via Google Web Speech API).")
        return True

    def transcribe(self, audio_source: Union[str, bytes], language: str = "en-IN") -> str:
        """
        Transcribes WAV audio bytes (or a WAV file path) using Google Web Speech API.

        :param audio_source: Raw WAV bytes or a path string to a WAV file.
        :param language: BCP-47 language hint.  When "en-IN" is passed, the bilingual
                         en-IN → hi-IN fallback pipeline is activated automatically.
        :return: Transcribed text string, or "" on failure / silence.
        """
        if not audio_source:
            return ""

        try:
            # Accept both bytes (from mic stream) and file paths
            if isinstance(audio_source, (bytes, bytearray)):
                wav_io = io.BytesIO(audio_source)
                with sr.AudioFile(wav_io) as source:
                    audio_data = self._recognizer.record(source)
            else:
                with sr.AudioFile(str(audio_source)) as source:
                    audio_data = self._recognizer.record(source)
        except Exception as exc:
            logger.error("GoogleSpeechEngine: failed to read audio source — %s", exc)
            return ""

        return self._transcribe_audio(audio_data, language)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _transcribe_audio(self, audio_data: sr.AudioData, language: str) -> str:
        """Run the bilingual en-IN → hi-IN pipeline on an AudioData object."""
        try:
            if language == "en-IN":
                return self._en_in_with_fallback(audio_data)
            else:
                # Caller asked for a specific language — honour it directly
                return self._recognizer.recognize_google(audio_data, language=language)

        except sr.UnknownValueError:
            return ""
        except sr.RequestError as exc:
            logger.error("Google STT RequestError: %s", exc)
            return ""
        except Exception as exc:
            logger.error("Google STT unexpected error: %s", exc)
            return ""

    def _en_in_with_fallback(self, audio_data: sr.AudioData) -> str:
        """
        Primary en-IN recognition with automatic hi-IN fallback.
        Returns the best transcript string.
        """
        try:
            result = self._recognizer.recognize_google(
                audio_data, language="en-IN", show_all=True
            )
            if isinstance(result, dict) and result.get("alternative"):
                best = result["alternative"][0]
                transcript = best.get("transcript", "")
                confidence = best.get("confidence", 1.0)

                if confidence < self.CONFIDENCE_THRESHOLD:
                    logger.info(
                        "Google STT en-IN confidence %.2f < %.2f — falling back to hi-IN.",
                        confidence, self.CONFIDENCE_THRESHOLD
                    )
                    return self._hi_in_fallback(audio_data) or transcript

                return transcript

            # No structured alternatives returned — try simple hi-IN pass
            return self._hi_in_fallback(audio_data)

        except sr.UnknownValueError:
            # en-IN completely failed — try Hindi
            return self._hi_in_fallback(audio_data)
        except sr.RequestError:
            raise  # Let caller handle network errors

    def _hi_in_fallback(self, audio_data: sr.AudioData) -> str:
        """Secondary pass using hi-IN locale."""
        try:
            return self._recognizer.recognize_google(audio_data, language="hi-IN")
        except (sr.UnknownValueError, sr.RequestError):
            return ""


# ---------------------------------------------------------------------------
# SpeechToText — public orchestration class
# ---------------------------------------------------------------------------

class SpeechToText:
    """
    Public STT orchestrator used by the rest of the Morphix application.
    Delegates all transcription to GoogleSpeechEngine by default.
    """

    def __init__(self, engine: Optional[SpeechEngine] = None) -> None:
        """
        :param engine: Optional custom SpeechEngine implementation (default: GoogleSpeechEngine).
        """
        self._listening: bool = False
        self._last_transcript: str = ""
        self.sample_rate: int = 16000

        self.engine: SpeechEngine = engine if engine else GoogleSpeechEngine()
        self.initialize()

    def initialize(self) -> bool:
        """Initialises the underlying speech engine."""
        return self.engine.initialize()

    def start_listening(self) -> None:
        """Enables listening state flag."""
        self._listening = True

    def stop_listening(self) -> None:
        """Disables listening state flag."""
        self._listening = False

    def is_listening(self) -> bool:
        """Returns whether the STT processor is active."""
        return self._listening

    def create_recognizer(self) -> Optional[Any]:
        """No-op — kept for backward compatibility."""
        return None

    def transcribe(self, audio_source: Union[str, bytes], language: str = "en-IN") -> str:
        """
        Transcribes WAV audio (bytes or file path) via the configured engine.

        :param audio_source: Raw WAV bytes or a WAV file path string.
        :param language: BCP-47 language tag passed to the engine.
        :return: Transcribed text string.
        """
        if not audio_source:
            return ""

        import inspect
        try:
            sig = inspect.signature(self.engine.transcribe)
            if "language" in sig.parameters:
                transcript = self.engine.transcribe(audio_source, language=language)
            else:
                transcript = getattr(self.engine, "transcribe")(audio_source)
        except Exception:
            transcript = self.engine.transcribe(audio_source, language=language)

        self._last_transcript = transcript
        return transcript

    def get_last_transcript(self) -> str:
        """Returns the most recently transcribed text."""
        return self._last_transcript

    def clear_last_transcript(self) -> None:
        """Clears the last transcribed text."""
        self._last_transcript = ""

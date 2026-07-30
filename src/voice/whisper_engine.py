"""
Morphix Desktop Companion Robot - Whisper.cpp Speech Engine
============================================================
File: src/voice/whisper_engine.py
Description:
    High-performance Speech-to-Text backend powered by Whisper.cpp (via pywhispercpp).
    Implements SpeechEngine interface, restricts downloads, loads the model exactly once,
    and returns complete transcription sentences.
"""

from contextlib import contextmanager
import logging
import os
import re
import sys
import tempfile
import time
from typing import Optional, Union

from config import Config
from src.voice.speech_engine import SpeechEngine

# Setup logger for WhisperEngine module
logger = logging.getLogger("Morphix.WhisperEngine")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [WhisperEngine]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull to suppress native C++ logging."""
    devnull = None
    saved_stdout_fd = None
    saved_stderr_fd = None
    try:
        devnull = os.open(os.devnull, os.O_RDWR)
        saved_stdout_fd = os.dup(1)
        saved_stderr_fd = os.dup(2)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    except Exception:
        yield
    finally:
        if devnull is not None:
            try:
                if saved_stdout_fd is not None:
                    os.dup2(saved_stdout_fd, 1)
                    os.close(saved_stdout_fd)
                if saved_stderr_fd is not None:
                    os.dup2(saved_stderr_fd, 2)
                    os.close(saved_stderr_fd)
                os.close(devnull)
            except Exception:
                pass


class WhisperEngine(SpeechEngine):
    """
    Whisper.cpp implementation of SpeechEngine using pywhispercpp.
    """

    # Class-level model singleton to ensure Whisper is loaded only once
    _model = None
    _loaded_model_name: Optional[str] = None

    def __init__(self, model_name: Optional[str] = None) -> None:
        """
        Initializes the WhisperEngine instance.
        """
        self.model_name: str = model_name or getattr(Config, "STT_MODEL", "tiny.en")
        self.models_dir: str = getattr(Config, "WHISPER_MODELS_DIR", "models/whisper")

    def initialize(self) -> bool:
        """
        Initializes the Whisper.cpp model.
        Strictly checks for the model file on disk and fails if missing.
        """
        if WhisperEngine._model is not None:
            return True

        # Construct path to the expected model file
        model_filename = f"ggml-{self.model_name}.bin"
        model_filepath = os.path.join(self.models_dir, model_filename)

        # Normalize and resolve path
        resolved_path = os.path.abspath(model_filepath)

        if not os.path.exists(resolved_path):
            print(f"\nRequired model not found:\n{model_filepath}\n")
            print("Please download the model manually.\n")
            logger.error("Required model file missing at: %s", resolved_path)
            sys.exit(1)

        try:
            logger.info("Initializing Whisper model '%s' from '%s'...", self.model_name, resolved_path)
            start_time = time.time()

            from pywhispercpp.model import Model
            WhisperEngine._model = Model(model=self.model_name, models_dir=self.models_dir)
            WhisperEngine._loaded_model_name = self.model_name

            duration = time.time() - start_time
            print("Whisper model loaded")
            logger.info("Whisper model '%s' loaded in %.2f seconds.", self.model_name, duration)
            return True

        except Exception as e:
            logger.error("Failed to load Whisper.cpp model: %s", str(e))
            return False

    def transcribe(self, audio_source: Union[str, bytes]) -> str:
        """
        Transcribes 16kHz WAV audio using Whisper.cpp.
        """
        if WhisperEngine._model is None:
            if not self.initialize():
                return ""

        temp_wav_path = None
        # Handle raw WAV bytes by writing them to a temporary file
        if isinstance(audio_source, bytes):
            try:
                os.makedirs("temp/audio", exist_ok=True)
                with tempfile.NamedTemporaryFile(suffix=".wav", dir="temp/audio", delete=False) as f:
                    f.write(audio_source)
                    temp_wav_path = f.name
                audio_path = temp_wav_path
            except Exception as e:
                logger.error("Failed to write temporary WAV file for transcription: %s", str(e))
                return ""
        else:
            audio_path = audio_source

        if not os.path.exists(audio_path):
            logger.error("Audio file does not exist: %s", audio_path)
            return ""

        try:
            start_time = time.time()
            with suppress_stdout_stderr():
                segments = WhisperEngine._model.transcribe(audio_path, n_processors=4)
            duration = time.time() - start_time

            raw_sentence = " ".join([seg.text.strip() for seg in segments]).strip()
            
            # Clean internal blank audio tokens, repeated spaces, and leading/trailing whitespace
            cleaned_sentence = re.sub(r'(?i)\[BLANK_AUDIO\]', '', raw_sentence)
            cleaned_sentence = re.sub(r'\s+', ' ', cleaned_sentence).strip()
            final_sentence = cleaned_sentence

            print(f"Final transcript: {final_sentence}")
            print(f"Transcription time: {duration:.2f}s")

            logger.info("Whisper final transcript: '%s'", final_sentence)
            logger.info("Whisper transcription duration: %.2f seconds", duration)

            return final_sentence

        except Exception as e:
            logger.error("Error during Whisper.cpp transcription: %s", str(e))
            return ""
        finally:
            if temp_wav_path and os.path.exists(temp_wav_path):
                try:
                    os.remove(temp_wav_path)
                except Exception:
                    pass

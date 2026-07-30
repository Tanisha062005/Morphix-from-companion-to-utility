"""
Morphix Desktop Companion Robot - Wake Word Manager
=====================================================
File: src/wakeword/wakeword_manager.py
Description:
    High-level manager for Morphix wake word detection.
    Encapsulates backend wake word engines (OpenWakeWord, Porcupine, Custom TFLite, Fallback)
    and provides a clean, unified interface for VoiceAssistant following SOLID principles.
"""

import logging
from pathlib import Path
import sys
from typing import Optional, Union

from src.wakeword.wakeword_engine import BaseWakeWordEngine, FallbackWakeWordEngine, OpenWakeWordEngine

# Setup logger for WakeWordManager module
logger = logging.getLogger("Morphix.WakeWordManager")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [WakeWordManager]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class WakeWordManager:
    """
    Coordinator and manager for wake word engines in Morphix.
    Enables swappable engine backends via Dependency Injection.
    """

    def __init__(self, engine: Optional[BaseWakeWordEngine] = None) -> None:
        """
        Initializes the WakeWordManager with a specific wake word engine backend.

        :param engine: Instance of BaseWakeWordEngine (default: OpenWakeWordEngine).
        """
        self.engine: BaseWakeWordEngine = engine if engine else OpenWakeWordEngine()
        self._listening: bool = False

        logger.info("WakeWordManager initialized with primary engine backend: %s", self.engine.get_engine_name())

    def initialize(self) -> bool:
        """
        Initializes the underlying wake word engine backend.
        If custom OpenWakeWord model is missing from models/wakeword/, automatically
        switches to FallbackEngine so live microphone testing works immediately without crashing.

        :return: True if initialized successfully with an active engine.
        """
        success = self.engine.initialize()

        if not success and isinstance(self.engine, OpenWakeWordEngine):
            logger.warning("Custom Morphix wake-word model not found in models/wakeword/.")
            logger.info("Enabling Fallback Engine for immediate live microphone testing ('Hey Morphix').")
            self.engine = FallbackWakeWordEngine(target_phrase="Hey Morphix")
            success = self.engine.initialize()

        return success

    def start(self) -> None:
        """
        Enables the wake word detection listening state.
        """
        self._listening = True
        self.engine.start()
        logger.info("WakeWordManager listening state enabled (Engine: %s).", self.engine.get_engine_name())

    def stop(self) -> None:
        """
        Disables the wake word detection listening state.
        """
        self._listening = False
        self.engine.stop()
        logger.info("WakeWordManager listening state disabled.")

    def is_listening(self) -> bool:
        """
        Returns whether the manager is currently listening for a wake word.

        :return: True if listening, False otherwise.
        """
        return self._listening

    def is_detected(self, audio_frame: Optional[bytes] = None) -> bool:
        """
        Evaluates whether the wake word is detected by the active engine.

        :param audio_frame: Optional raw PCM audio bytes or chunk.
        :return: True if wake word is detected, False otherwise.
        """
        if not self._listening:
            return False

        return self.engine.is_detected(audio_frame)

    def get_engine_name(self) -> str:
        """
        Returns the name of the currently active wake word engine backend.

        :return: Engine name string.
        """
        return self.engine.get_engine_name()

    def get_model_path(self) -> Optional[Path]:
        """
        Returns the model Path from the active engine backend.

        :return: Path or None.
        """
        return self.engine.get_model_path()

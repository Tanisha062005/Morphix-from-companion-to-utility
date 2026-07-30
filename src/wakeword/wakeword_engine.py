"""
Morphix Desktop Companion Robot - Modular Wake Word Engine
===========================================================
File: src/wakeword/wakeword_engine.py
Description:
    Provides a modular, swappable Wake Word Engine architecture for Morphix.
    Defines BaseWakeWordEngine interface, OpenWakeWordEngine (designed for
    models/wakeword/morphix.tflite), and FallbackWakeWordEngine for ready-to-use testing.
"""

from abc import ABC, abstractmethod
import logging
from pathlib import Path
import string
import sys
from typing import Optional, Union

# Setup logger for WakeWordEngine module
logger = logging.getLogger("Morphix.WakeWordEngine")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [WakeWordEngine]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class BaseWakeWordEngine(ABC):
    """
    Abstract Base Class defining the unified interface for all wake word engine backends.
    """

    @abstractmethod
    def initialize(self) -> bool:
        """
        Initializes engine resources and model files.

        :return: True if successfully initialized, False otherwise.
        """
        pass

    @abstractmethod
    def start(self) -> None:
        """
        Enables wake word detection listening state.
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """
        Disables wake word detection listening state.
        """
        pass

    @abstractmethod
    def is_detected(self, audio_frame: Optional[bytes] = None) -> bool:
        """
        Evaluates an audio frame or input chunk for wake word occurrence.

        :param audio_frame: Raw PCM audio bytes or audio data chunk.
        :return: True if wake word is detected, False otherwise.
        """
        pass

    @abstractmethod
    def get_model_path(self) -> Optional[Path]:
        """
        Returns the resolved Path to the wake word model file.

        :return: Path object or None if missing.
        """
        pass

    @abstractmethod
    def get_engine_name(self) -> str:
        """
        Returns the human-readable backend engine name.

        :return: Name string.
        """
        pass


class OpenWakeWordEngine(BaseWakeWordEngine):
    """
    OpenWakeWord engine implementation for Morphix.
    Loads custom TFLite or ONNX models from models/wakeword/.
    Handles missing custom models gracefully with clear logging.
    """

    def __init__(self, model_dir: Optional[Union[str, Path]] = None, threshold: float = 0.5) -> None:
        """
        Initializes OpenWakeWordEngine with target model directory.

        :param model_dir: Path to custom models directory (default: models/wakeword/).
        :param threshold: Detection confidence threshold (0.0 to 1.0).
        """
        self.project_root: Path = self._find_project_root()
        if model_dir:
            self.model_dir: Path = Path(model_dir).resolve()
        else:
            self.model_dir: Path = self.project_root / "models" / "wakeword"

        self.threshold: float = threshold
        self._active: bool = False
        self._initialized: bool = False
        self._model_path: Optional[Path] = None
        self._model_instance = None

        self._ensure_model_dir()

    def _find_project_root(self) -> Path:
        """
        Locates project root directory using pathlib.

        :return: Path object referencing project root.
        """
        current_file = Path(__file__).resolve()
        for parent in current_file.parents:
            if (parent / "src").exists() or (parent / "config.py").exists():
                return parent
        return current_file.parents[2]

    def _ensure_model_dir(self) -> None:
        """
        Ensures the target models/wakeword directory exists on disk.
        """
        try:
            self.model_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error("Failed to create wakeword models directory at %s: %s", self.model_dir, str(e))

    def _find_custom_model_file(self) -> Optional[Path]:
        """
        Searches models/wakeword for custom Morphix model files (*.tflite, *.onnx).

        :return: Path to custom model file if present, None otherwise.
        """
        if not self.model_dir.exists():
            return None

        expected_names = ["morphix.tflite", "morphix.onnx"]
        for name in expected_names:
            candidate = self.model_dir / name
            if candidate.exists() and candidate.is_file():
                return candidate

        candidates = list(self.model_dir.glob("*.tflite")) + list(self.model_dir.glob("*.onnx"))
        if candidates:
            return candidates[0]

        return None

    def initialize(self) -> bool:
        """
        Initializes OpenWakeWord engine with custom model from models/wakeword/.

        :return: True if successfully loaded, False if model missing.
        """
        self._model_path = self._find_custom_model_file()

        if not self._model_path or not self._model_path.exists():
            logger.warning("Custom Morphix wake-word model not found at %s", self.model_dir)
            logger.info("To use custom OpenWakeWord detection, place 'morphix.tflite' inside models/wakeword/.")
            self._initialized = False
            return False

        try:
            import openwakeword
            logger.info("Loading custom OpenWakeWord model from: %s", self._model_path)
            self._model_instance = openwakeword.Model(
                wakeword_models=[str(self._model_path)],
                inference_framework="onnx" if self._model_path.suffix == ".onnx" else "tflite"
            )
            self._initialized = True
            logger.info("OpenWakeWord engine initialized successfully.")
            return True

        except ImportError:
            logger.error("openwakeword module is not installed. Unable to load custom model.")
            self._initialized = False
            return False
        except Exception as e:
            logger.error("Failed to load custom wake-word model from %s: %s", self._model_path, str(e))
            self._initialized = False
            return False

    def start(self) -> None:
        """Enables detection listening state."""
        self._active = True

    def stop(self) -> None:
        """Disables detection listening state."""
        self._active = False

    def is_detected(self, audio_frame: Optional[bytes] = None) -> bool:
        """
        Evaluates audio frame using OpenWakeWord model.

        :param audio_frame: Raw 16kHz 16-bit PCM audio frame bytes.
        :return: True if model score exceeds threshold, False otherwise.
        """
        if not self._active or not self._initialized or not self._model_instance or not audio_frame:
            return False

        try:
            import numpy as np
            pcm_array = np.frombuffer(audio_frame, dtype=np.int16)
            prediction = self._model_instance.predict(pcm_array)

            for model_name, scores in prediction.items():
                score = scores[-1] if isinstance(scores, (list, np.ndarray)) else float(scores)
                if score >= self.threshold:
                    logger.info("Wake word detected by OpenWakeWord model '%s' (score: %.3f).", model_name, score)
                    return True

            return False

        except Exception as e:
            logger.error("Error evaluating audio frame in OpenWakeWordEngine: %s", str(e))
            return False

    def get_model_path(self) -> Optional[Path]:
        return self._model_path

    def get_engine_name(self) -> str:
        return "OpenWakeWord"


class FallbackWakeWordEngine(BaseWakeWordEngine):
    """
    Fallback acoustic/Vosk wake-word engine used when no custom TFLite model exists in models/wakeword/.
    Allows immediate live microphone testing of "Hey Morphix".
    """

    def __init__(self, target_phrase: str = "Hey Morphix") -> None:
        self.target_phrase: str = target_phrase
        self._active: bool = False
        self._initialized: bool = False
        self._stt = None

    def initialize(self) -> bool:
        try:
            from src.voice.speech_to_text import SpeechToText
            self._stt = SpeechToText()
            self._initialized = True
            logger.info("FallbackWakeWordEngine initialized for immediate live testing ('%s').", self.target_phrase)
            return True
        except Exception as e:
            logger.error("Failed to initialize FallbackWakeWordEngine: %s", str(e))
            return False

    def start(self) -> None:
        self._active = True

    def stop(self) -> None:
        self._active = False

    def _normalize(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        lowercased = text.lower()
        no_punct = lowercased.translate(str.maketrans("", "", string.punctuation))
        return " ".join(no_punct.split()).strip()

    def is_detected(self, audio_frame: Optional[bytes] = None) -> bool:
        if not self._active or not self._initialized or not audio_frame or not self._stt:
            return False

        try:
            from src.voice.wake_word import WakeWordDetector
            detector = WakeWordDetector(wake_word=self.target_phrase, threshold=0.70)

            transcript = self._stt.transcribe(audio_frame)
            if not transcript:
                return False

            if detector.detect(transcript):
                logger.info("Wake word '%s' detected by Fallback Engine (Fuzzy Match)!", self.target_phrase)
                return True

            return False

        except Exception as e:
            logger.error("Error in FallbackWakeWordEngine detection: %s", str(e))
            return False


    def get_model_path(self) -> Optional[Path]:
        return None

    def get_engine_name(self) -> str:
        return "FallbackEngine (Live Testing)"

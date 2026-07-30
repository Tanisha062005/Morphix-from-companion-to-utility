"""
Morphix Desktop Companion Robot - Text-To-Speech (Piper Engine)
================================================================
File: src/voice/text_to_speech.py
Description:
    Production-ready offline Text-to-Speech (TTS) module utilizing 
    the official Piper binary executable via subprocess.

    Automatically detects project dependencies (piper.exe inside project_root/piper/,
    ONNX models & JSON configs inside project_root/models/piper/ or project_root/src/models/piper/),
    verifies paths before speaking, logs detected paths at startup, and plays speech audio.
"""

import logging
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Optional

# Setup logger for TTS module
logger = logging.getLogger("Morphix.TTS")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [TTS]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class TextToSpeech:
    """
    Offline Text-to-Speech engine utilizing official Piper executable.
    Handles binary/model auto-discovery, path verification, speech synthesis, and audio playback.
    """

    def __init__(
        self,
        piper_path: Optional[str] = None,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None
    ) -> None:
        """
        Initializes the TextToSpeech engine and locates dependencies.

        :param piper_path: Optional explicit path to piper executable.
        :param model_path: Optional explicit path to ONNX voice model.
        :param config_path: Optional explicit path to ONNX model JSON config.
        """
        self.project_root: Path = self._find_project_root()
        self.temp_audio_dir: Path = self.project_root / "temp" / "audio"
        self.temp_audio_dir.mkdir(parents=True, exist_ok=True)

        self.piper_exe: Optional[Path] = Path(piper_path) if piper_path else self._find_piper_exe()
        self.model_path: Optional[Path] = Path(model_path) if model_path else self._find_model_file()
        self.config_path: Optional[Path] = Path(config_path) if config_path else self._find_config_file()

        self._play_process: Optional[subprocess.Popen] = None
        self._is_speaking: bool = False

        # Print/log detected paths when application starts
        self._log_detected_paths()

        # Verify all paths
        self._validate_environment()

    def _find_project_root(self) -> Path:
        """
        Locates the project root directory using pathlib.

        :return: Path object pointing to project root directory.
        """
        current_file = Path(__file__).resolve()
        for parent in current_file.parents:
            if (parent / "src").exists() or (parent / "config.py").exists() or (parent / "piper").exists():
                return parent
        return current_file.parents[2]

    def _find_piper_exe(self) -> Optional[Path]:
        """
        Searches candidate locations for the piper binary executable.
        Prioritizes project_root/piper/piper.exe.

        :return: Path to piper executable if found, None otherwise.
        """
        exe_name = "piper.exe" if platform.system() == "Windows" else "piper"

        # List candidate paths relative to project root (prioritize project_root/piper/piper.exe)
        candidates = [
            self.project_root / "piper" / exe_name,
            self.project_root / "src" / "piper" / exe_name,
            self.project_root / "models" / "piper" / exe_name,
            self.project_root / "src" / "models" / "piper" / exe_name,
        ]

        for path in candidates:
            if path.exists() and path.is_file():
                return path

        # Search project directory recursively for piper executable
        found_files = list(self.project_root.rglob(exe_name))
        if found_files:
            return found_files[0]

        # Check system PATH
        system_piper = shutil.which(exe_name) or shutil.which("piper")
        if system_piper:
            return Path(system_piper)

        return None

    def _find_model_file(self) -> Optional[Path]:
        """
        Searches candidate locations for ONNX voice model files (*.onnx).
        Prioritizes project_root/models/piper/ and project_root/src/models/piper/.

        :return: Path to ONNX model file if found, None otherwise.
        """
        search_dirs = [
            self.project_root / "models" / "piper",
            self.project_root / "src" / "models" / "piper",
            self.project_root / "piper",
        ]

        for s_dir in search_dirs:
            if s_dir.exists():
                onnx_files = [f for f in s_dir.glob("*.onnx") if not f.name.endswith(".onnx.json")]
                if onnx_files:
                    return onnx_files[0]

        # Recursive search fallback
        all_onnx = [f for f in self.project_root.rglob("*.onnx") if not f.name.endswith(".onnx.json")]
        if all_onnx:
            return all_onnx[0]

        return None

    def _find_config_file(self) -> Optional[Path]:
        """
        Searches for the matching JSON configuration file for the voice model.

        :return: Path to JSON config file if found, None otherwise.
        """
        if self.model_path:
            # Check for direct <model_name>.json
            expected_json = Path(str(self.model_path) + ".json")
            if expected_json.exists():
                return expected_json

            # Check directory for any .json file matching the model
            model_dir = self.model_path.parent
            json_files = list(model_dir.glob("*.json"))
            if json_files:
                return json_files[0]

        return None

    def _log_detected_paths(self) -> None:
        """
        Prints and logs all detected dependency paths at application startup.
        """
        logger.info("==================================================")
        logger.info("       MORPHIX TTS DEPENDENCY PATH DETECTION      ")
        logger.info("==================================================")
        logger.info("Project Root       : %s", self.project_root)
        logger.info("Piper Executable   : %s", self.piper_exe if self.piper_exe else "NOT FOUND")
        logger.info("Voice Model (.onnx): %s", self.model_path if self.model_path else "NOT FOUND")
        logger.info("Voice Config (.json): %s", self.config_path if self.config_path else "NOT FOUND")
        logger.info("Temp Audio Directory: %s", self.temp_audio_dir)
        logger.info("==================================================")

    def _validate_environment(self) -> bool:
        """
        Verifies all required file paths before speech synthesis.

        :return: True if environment and all paths are valid, False otherwise.
        """
        valid = True
        if not self.piper_exe or not self.piper_exe.exists() or not self.piper_exe.is_file():
            logger.error("Path Verification Failed: Piper binary ('piper.exe') does not exist at %s", self.piper_exe)
            valid = False
        if not self.model_path or not self.model_path.exists() or not self.model_path.is_file():
            logger.error("Path Verification Failed: ONNX voice model does not exist at %s", self.model_path)
            valid = False
        if self.config_path and (not self.config_path.exists() or not self.config_path.is_file()):
            logger.warning("Path Warning: Voice config (.json) specified but file does not exist at %s", self.config_path)

        return valid

    def save_to_file(self, text: str, output_path: str) -> bool:
        """
        Synthesizes text into speech and saves the resulting WAV file to disk.
        Verifies all paths prior to execution.

        :param text: Text string to convert to speech.
        :param output_path: File path where the output WAV audio will be saved.
        :return: True if audio file was successfully generated, False otherwise.
        """
        if not text or not text.strip():
            logger.warning("Cannot synthesize empty or whitespace text.")
            return False

        # Verify all paths before speaking
        if not self._validate_environment():
            logger.error("Speech synthesis aborted: invalid or missing paths.")
            return False

        out_file = Path(output_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(self.piper_exe),
            "--model", str(self.model_path),
            "--output_file", str(out_file)
        ]

        if self.config_path and self.config_path.exists():
            cmd.extend(["--config", str(self.config_path)])

        try:
            logger.info("Running Piper TTS synthesis for: '%s'", text)
            result = subprocess.run(
                cmd,
                input=text.strip().encode("utf-8"),
                capture_output=True,
                check=True
            )
            if out_file.exists() and out_file.stat().st_size > 0:
                logger.info("Successfully generated speech audio at: %s", out_file)
                return True
            else:
                logger.error("Piper execution finished, but output audio file is missing or empty.")
                return False

        except subprocess.CalledProcessError as e:
            logger.error("Piper execution error (returncode %d): %s", e.returncode, e.stderr.decode("utf-8", errors="ignore"))
            return False
        except Exception as e:
            logger.error("Unexpected error during speech synthesis: %s", str(e))
            return False

    def speak(self, text: str) -> bool:
        """
        Synthesizes speech from text and automatically plays it through system speaker.
        Verifies paths prior to generating audio.

        :param text: Text string to speak.
        :return: True if speech was generated and played successfully, False otherwise.
        """
        if not text or not text.strip():
            logger.warning("Cannot speak empty text.")
            return False

        # Verify paths before speech generation
        if not self._validate_environment():
            logger.error("Speech generation aborted: path verification failed.")
            return False

        # Generate temporary WAV file inside temp/audio/
        temp_wav = self.temp_audio_dir / f"speech_{int(time.time() * 1000)}.wav"

        success = self.save_to_file(text, str(temp_wav))
        if not success or not temp_wav.exists():
            return False

        try:
            self._is_speaking = True
            logger.info("Playing audio output...")

            if platform.system() == "Windows":
                import winsound
                # Play WAV synchronously using Windows winsound
                winsound.PlaySound(str(temp_wav), winsound.SND_FILENAME)
            else:
                # Fallback for Linux / Raspberry Pi using aplay / paplay
                player = shutil.which("aplay") or shutil.which("paplay") or shutil.which("ffplay")
                if player:
                    self._play_process = subprocess.Popen([player, str(temp_wav)])
                    self._play_process.wait()
                else:
                    logger.error("No suitable audio player binary (aplay/paplay) found on non-Windows system.")

            logger.info("Finished speaking.")
            return True

        except Exception as e:
            logger.error("Audio playback error: %s", str(e))
            return False
        finally:
            self._is_speaking = False
            # Clean up temporary WAV file safely
            try:
                if temp_wav.exists():
                    temp_wav.unlink()
            except Exception:
                pass

    def stop(self) -> None:
        """
        Stops active audio playback immediately.
        """
        logger.info("Stopping audio playback...")
        self._is_speaking = False

        if platform.system() == "Windows":
            try:
                import winsound
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception as e:
                logger.error("Error purging winsound playback: %s", str(e))

        if self._play_process and self._play_process.poll() is None:
            try:
                self._play_process.terminate()
                self._play_process = None
            except Exception as e:
                logger.error("Error terminating play process: %s", str(e))

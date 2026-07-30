"""
Morphix Desktop Companion Robot - Sound Player
================================================
File: src/audio/sound_player.py
Description:
    Provides cross-platform, blocking WAV audio playback using pygame.mixer.
    Automatically manages asset directory structures, verifies audio file paths,
    and handles missing files or playback exceptions gracefully without crashing.
"""

import logging
from pathlib import Path
import sys
from typing import Optional, Union

# Setup logger for SoundPlayer module
logger = logging.getLogger("Morphix.SoundPlayer")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [SoundPlayer]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class SoundPlayer:
    """
    Manages cross-platform audio playback using pygame.mixer.
    Handles file verification, blocking audio playback, and asset directory creation.
    """

    def __init__(self, sounds_dir: Optional[Union[str, Path]] = None) -> None:
        """
        Initializes the SoundPlayer instance and ensures target sounds directory exists.

        :param sounds_dir: Optional custom path to the sounds directory.
                           Defaults to project_root/assets/sounds/.
        """
        self.project_root: Path = self._find_project_root()
        if sounds_dir:
            self.sounds_dir: Path = Path(sounds_dir).resolve()
        else:
            self.sounds_dir: Path = self.project_root / "assets" / "sounds"

        # Automatically create assets/sounds/ directory if it does not exist
        self._ensure_sounds_directory()

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

    def _ensure_sounds_directory(self) -> None:
        """
        Creates the target sounds directory if it does not already exist.
        """
        try:
            self.sounds_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Sounds directory verified at: %s", self.sounds_dir)
        except Exception as e:
            logger.error("Failed to create sounds directory at %s: %s", self.sounds_dir, str(e))

    def get_sounds_dir(self) -> Path:
        """
        Returns the resolved Path to the sounds directory.

        :return: Path object for sounds directory.
        """
        return self.sounds_dir

    def play_sound(self, sound_file: Union[str, Path]) -> bool:
        """
        Plays a WAV audio file synchronously (blocking until playback completes).
        Gracefully handles missing files or playback failures.

        :param sound_file: Path or filename of the WAV file to play.
        :return: True if audio was played successfully, False otherwise.
        """
        file_path = Path(sound_file)
        if not file_path.is_absolute():
            # If relative path, look inside sounds_dir first, then relative to working directory
            candidate_path = self.sounds_dir / file_path
            if candidate_path.exists():
                file_path = candidate_path
            else:
                file_path = file_path.resolve()

        # Handle missing files gracefully
        if not file_path.exists() or not file_path.is_file():
            logger.info("Audio file not found at '%s'. Skipping playback gracefully.", file_path)
            return False

        # Attempt audio playback using pygame.mixer
        try:
            import pygame

            # Safely initialize pygame mixer
            if not pygame.mixer.get_init():
                pygame.mixer.init()

            logger.info("Starting playback for audio file: %s", file_path)
            sound = pygame.mixer.Sound(str(file_path))
            channel = sound.play()

            # Blocking wait until playback finishes
            while channel and channel.get_busy():
                pygame.time.wait(30)

            logger.info("Completed playback for audio file: %s", file_path)
            return True

        except ImportError:
            logger.error("pygame module is not installed. Unable to play audio file '%s'.", file_path)
            return False
        except Exception as e:
            logger.error("Error occurred while playing sound '%s': %s", file_path, str(e))
            return False
        finally:
            # Safely shut down pygame.mixer after playback attempt
            try:
                import pygame
                if pygame.mixer.get_init():
                    pygame.mixer.quit()
            except Exception:
                pass

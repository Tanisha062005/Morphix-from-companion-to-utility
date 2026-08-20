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

        self.preloaded_sounds = {}
        self.bark_channel = None
        self._init_mixer_and_preload()

    def _init_mixer_and_preload(self) -> None:
        """
        Pre-initializes the pygame mixer once and preloads morphix_voice.wav into memory.
        """
        try:
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            logger.info("pygame.mixer initialized successfully at startup.")

            # Preload morphix_voice.wav
            wake_sound_name = "morphix_voice.wav"
            candidates = [
                self.sounds_dir / wake_sound_name,
                self.project_root / "assets" / "sounds" / wake_sound_name,
                Path("assets/sounds") / wake_sound_name,
                Path("assets/sounds/morphix_voice.wav")
            ]
            for p in candidates:
                resolved = p.resolve()
                if resolved.exists() and resolved.is_file():
                    sound = pygame.mixer.Sound(str(resolved))
                    # Store multiple formats of the key for robust lookups
                    self.preloaded_sounds[wake_sound_name] = sound
                    self.preloaded_sounds[str(resolved)] = sound
                    self.preloaded_sounds[str(p)] = sound
                    self.preloaded_sounds["assets/sounds/morphix_voice.wav"] = sound
                    logger.info("Pre-loaded sound asset: %s", resolved)
                    break
            else:
                logger.warning("Could not locate %s for pre-loading at startup.", wake_sound_name)
        except ImportError:
            logger.error("pygame module is not installed. Audio pre-loading and playback will be unavailable.")
        except Exception as e:
            logger.error("Error during pygame mixer initialization or sound pre-loading: %s", str(e))

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

    def play_sound(self, sound_file: Union[str, Path], blocking: bool = True) -> bool:
        """
        Plays a WAV audio file. Uses preloaded sound in memory if available.
        Can play synchronously (blocking) or asynchronously (non-blocking).

        :param sound_file: Path or filename of the WAV file to play.
        :param blocking: If True, blocks execution until playback completes.
                         If False, plays in the background and returns immediately.
        :return: True if audio was played successfully, False otherwise.
        """
        file_path = Path(sound_file)
        file_str = str(sound_file)

        # Check if preloaded
        sound = None
        for key in [file_str, file_path.name, str(file_path.resolve())]:
            if key in self.preloaded_sounds:
                sound = self.preloaded_sounds[key]
                break

        # Attempt to play
        try:
            import pygame

            # Ensure mixer is initialized
            if not pygame.mixer.get_init():
                pygame.mixer.init()

            if sound is None:
                if not file_path.is_absolute():
                    candidate_path = self.sounds_dir / file_path
                    if candidate_path.exists():
                        file_path = candidate_path
                    else:
                        file_path = file_path.resolve()

                if not file_path.exists() or not file_path.is_file():
                    logger.info("Audio file not found at '%s'. Skipping playback gracefully.", file_path)
                    return False

                logger.info("Loading audio file dynamically: %s", file_path)
                sound = pygame.mixer.Sound(str(file_path))

            logger.info("Starting playback for audio file: %s (blocking=%s)", sound_file, blocking)
            channel = sound.play()

            # Keep track of the last played channel in case we need channel-specific check
            if "morphix_voice.wav" in file_str or file_path.name == "morphix_voice.wav":
                self.bark_channel = channel

            if blocking:
                # Blocking wait until playback finishes
                while channel and channel.get_busy():
                    pygame.time.wait(30)
                logger.info("Completed blocking playback for audio file: %s", sound_file)

            return True

        except ImportError:
            logger.error("pygame module is not installed. Unable to play audio file '%s'.", sound_file)
            return False
        except Exception as e:
            logger.error("Error occurred while playing sound '%s': %s", sound_file, str(e))
            return False

    def is_playing(self) -> bool:
        """
        Checks if the pygame mixer is currently playing any audio.
        """
        try:
            import pygame
            if pygame.mixer.get_init():
                # First check our registered bark channel
                if self.bark_channel and self.bark_channel.get_busy():
                    return True
                # Fallback to general mixer activity check
                return pygame.mixer.get_busy()
        except Exception:
            pass
        return False

"""
Morphix Desktop Companion Robot - OS & Digital Volume Controller
=================================================================
File: src/intent/volume_plugin.py
Description:
    Platform-adaptive master and digital volume controller supporting
    Windows (CoreAudio/ctypes/pygame) and Linux/Raspberry Pi (ALSA/PulseAudio).
"""

import ctypes
import logging
import os
import platform
import re
import subprocess
from typing import Any, Dict, Optional

logger = logging.getLogger("Volume")


class VolumeService:
    """Manages system and digital playback volume across Windows and Linux/RPi."""

    _instance: Optional["VolumeService"] = None

    def __init__(self) -> None:
        self.is_windows = platform.system().lower() == "windows"
        self.is_linux = platform.system().lower() == "linux"
        self._current_volume = 70  # default percentage (0-100)
        self._is_muted = False
        self._previous_volume = 70

        # Attempt to read current system volume on init
        self._init_volume()

    @classmethod
    def get_instance(cls) -> "VolumeService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _get_windows_endpoint(self):
        """Retrieves Windows CoreAudio EndpointVolume interface via pycaw."""
        try:
            from pycaw.pycaw import AudioUtilities
            speakers = AudioUtilities.GetSpeakers()
            if hasattr(speakers, "EndpointVolume") and speakers.EndpointVolume is not None:
                return speakers.EndpointVolume
            from ctypes import cast, POINTER
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import IAudioEndpointVolume
            interface = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            return cast(interface, POINTER(IAudioEndpointVolume))
        except Exception as ex:
            logger.debug("Could not get Windows endpoint volume: %s", ex)
            return None

    def _init_volume(self) -> None:
        """Initializes baseline volume level from system without resetting to 0%."""
        if self.is_windows:
            ep = self._get_windows_endpoint()
            if ep is not None:
                try:
                    # Check if muted
                    try:
                        is_muted = bool(ep.GetMute())
                    except Exception:
                        is_muted = False

                    scalar = ep.GetMasterVolumeLevelScalar()
                    if is_muted or scalar < 0.20:
                        logger.info("Master volume was low (%.2f) or muted (%s). Setting safe default 60%%.", scalar, is_muted)
                        ep.SetMute(0, None)
                        ep.SetMasterVolumeLevelScalar(0.60, None)
                        self._current_volume = 60
                        self._is_muted = False
                    else:
                        self._current_volume = int(round(scalar * 100))
                        self._is_muted = False
                        logger.info("Initialized volume from Windows Master: %d%%", self._current_volume)
                    return
                except Exception as ex:
                    logger.debug("Failed initializing Windows endpoint volume: %s", ex)

        # Fallback default if not on Windows or endpoint failed
        self._current_volume = 60
        self._is_muted = False
        try:
            import pygame
            if pygame.mixer.get_init():
                pygame.mixer.music.set_volume(0.60)
        except Exception:
            pass

    def get_volume(self) -> int:
        """Returns the current volume percentage (0-100)."""
        if self.is_windows:
            ep = self._get_windows_endpoint()
            if ep is not None:
                try:
                    self._current_volume = int(round(ep.GetMasterVolumeLevelScalar() * 100))
                except Exception:
                    pass
        return self._current_volume

    def set_volume(self, level: int) -> int:
        """
        Sets master OS / application volume to a specified percentage (0-100).
        Clamps to valid bounds and returns the actual set volume.
        """
        level = max(0, min(100, int(level)))
        self._current_volume = level
        self._is_muted = (level == 0)

        # 1. Windows OS Master Volume Adjustment
        if self.is_windows:
            ep = self._get_windows_endpoint()
            if ep is not None:
                try:
                    ep.SetMasterVolumeLevelScalar(level / 100.0, None)
                    logger.info("Set Windows master volume via pycaw: %d%%", level)
                except Exception as ex:
                    logger.debug("Failed setting Windows scalar: %s", ex)

        # 2. Linux / Raspberry Pi ALSA & PulseAudio Adjustment
        elif self.is_linux:
            try:
                subprocess.run(
                    ["amixer", "sset", "Master", f"{level}%"],
                    check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                subprocess.run(
                    ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{level}%"],
                    check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                logger.info("Set Linux/RPi volume via ALSA/pactl: %d%%", level)
            except Exception as ex:
                logger.warning("Failed to adjust Linux OS volume: %s", ex)

        # 3. Update pygame digital volume
        try:
            import pygame
            if pygame.mixer.get_init():
                pygame.mixer.music.set_volume(level / 100.0)
        except Exception:
            pass

        logger.info("Volume updated to: %d%%", level)
        return level

    def increase_volume(self, step: int = 15) -> int:
        """Increases volume by step percentage."""
        return self.set_volume(self.get_volume() + step)

    def decrease_volume(self, step: int = 15) -> int:
        """Decreases volume by step percentage."""
        return self.set_volume(self.get_volume() - step)

    def mute(self) -> bool:
        """Mutes audio playback."""
        if not self._is_muted:
            self._previous_volume = self.get_volume()
            if self.is_windows:
                ep = self._get_windows_endpoint()
                if ep is not None:
                    try:
                        ep.SetMute(1, None)
                    except Exception:
                        pass
            self.set_volume(0)
            self._is_muted = True
        return True

    def unmute(self) -> bool:
        """Unmutes and restores previous volume level."""
        if self._is_muted:
            restore_vol = self._previous_volume if self._previous_volume > 0 else 50
            if self.is_windows:
                ep = self._get_windows_endpoint()
                if ep is not None:
                    try:
                        ep.SetMute(0, None)
                    except Exception:
                        pass
            self.set_volume(restore_vol)
            self._is_muted = False
        return True

    def format_speech_response(self, action: str, level: int = 70, is_hindi: bool = False) -> str:
        """Generates natural bilingual speech confirmation."""
        if action == "set":
            if is_hindi:
                return f"Volume {level} percent set kar diya."
            return f"Volume set to {level} percent."
        elif action in ("increase", "up"):
            if is_hindi:
                return f"Volume badha diya, ab {level} percent hai."
            return f"Volume increased to {level} percent."
        elif action in ("decrease", "down"):
            if is_hindi:
                return f"Volume kam kar diya, ab {level} percent hai."
            return f"Volume decreased to {level} percent."
        elif action == "mute":
            if is_hindi:
                return "Mute kar diya."
            return "Volume muted."
        elif action == "unmute":
            if is_hindi:
                return f"Unmute kar diya, volume {level} percent hai."
            return f"Unmuted, volume is at {level} percent."
        return f"Volume is at {level} percent."


class VolumeIntentPlugin:
    """NLU plugin for detecting volume control commands."""

    def __init__(self, service: Optional[VolumeService] = None) -> None:
        self.service = service or VolumeService.get_instance()

    def match(self, command: str) -> Optional[Any]:
        """Classifies volume adjustments from natural voice phrases."""
        from src.intent.intent_classifier import IntentResult, IntentType

        c = command.strip().lower()
        is_hindi = bool(re.search(r'\b(badhao|kam|karo|awaaz|dheeme|tez|chup)\b', c))

        # 1. Mute
        if re.search(r'\b(mute|mute volume|be quiet|chup ho jao|awaz band)\b', c):
            return IntentResult(
                intent=IntentType.MUTE_VOLUME,
                confidence=0.95,
                parameters={"action": "mute", "is_hindi": is_hindi}
            )

        # 2. Unmute
        if re.search(r'\b(unmute|unmute volume|awaz chalu|bolna shuru karo)\b', c):
            return IntentResult(
                intent=IntentType.UNMUTE_VOLUME,
                confidence=0.95,
                parameters={"action": "unmute", "is_hindi": is_hindi}
            )

        # 3. Increase volume (volume badhao, turn it up, louder)
        if re.search(r'\b(volume up|increase volume|louder|turn it up|awaz badhao|volume badhao|sound badhao)\b', c):
            return IntentResult(
                intent=IntentType.INCREASE_VOLUME,
                confidence=0.95,
                parameters={"action": "increase", "step": 15, "is_hindi": is_hindi}
            )

        # 4. Decrease volume (volume kam karo, turn it down, quieter)
        if re.search(r'\b(volume down|decrease volume|lower volume|turn it down|quieter|awaz kam karo|volume kam karo|sound kam karo)\b', c):
            return IntentResult(
                intent=IntentType.DECREASE_VOLUME,
                confidence=0.95,
                parameters={"action": "decrease", "step": 15, "is_hindi": is_hindi}
            )

        # 5. Set explicit volume level (e.g. "volume 70%", "volume to 50", "set volume 80")
        set_match = re.search(r'\b(?:set\s+)?(?:volume|sound|awaz)\s+(?:to\s+)?(\d{1,3})\s*(?:%|percent)?\b', c)
        if not set_match:
            set_match = re.search(r'\b(\d{1,3})\s*(?:%|percent)\s*(?:volume|sound)\b', c)

        if set_match:
            try:
                target_level = int(set_match.group(1))
                if 0 <= target_level <= 100:
                    return IntentResult(
                        intent=IntentType.SET_VOLUME,
                        confidence=0.95,
                        parameters={"action": "set", "level": target_level, "is_hindi": is_hindi}
                    )
            except ValueError:
                pass

        return None

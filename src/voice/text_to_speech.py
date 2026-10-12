"""
Morphix Desktop Companion Robot - Text-To-Speech (Microsoft Edge TTS Engine)
=============================================================================
File: src/voice/text_to_speech.py
Description:
    Production-ready online Text-to-Speech (TTS) module utilizing
    Microsoft Edge TTS (`edge-tts`) with the Aarti Neural voice.

    Language Routing:
      - English / Hinglish  → en-IN-AartiNeural
      - Hindi (Devanagari)  → hi-IN-AartiNeural

    Settings:
      - rate  : +5%   (lively, conversational pace)
      - pitch : +2Hz  (warm, cheerful tone)

    Audio is streamed to a temporary MP3 file inside temp/audio/,
    played via SoundPlayer (blocking), then immediately deleted.
"""

import asyncio
import logging
import os
import ssl
import sys
import time
from pathlib import Path
from typing import Any, Optional

import edge_tts
import edge_tts.communicate as _edge_tts_comm

from src.audio.sound_player import SoundPlayer

# ---------------------------------------------------------------------------
# SSL Certificate Fix for Edge TTS on Windows
# ---------------------------------------------------------------------------
# edge_tts.communicate hardcodes its own _SSL_CTX at module import time.
# On some Windows systems (Python 3.14), Microsoft's speech.platform.bing.com
# certificate triggers "certificate is not yet valid" because the system trust
# store or Python's bundled CA chain is stale. We replace _SSL_CTX with a
# non-verifying context scoped strictly to edge_tts — no other SSL connections
# in the app are affected.
try:
    _patched_ssl_ctx = ssl.create_default_context()
    _patched_ssl_ctx.check_hostname = False
    _patched_ssl_ctx.verify_mode = ssl.CERT_NONE
    _edge_tts_comm._SSL_CTX = _patched_ssl_ctx
except Exception as _ssl_patch_err:
    pass  # If patching fails, edge_tts will use its original context

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger("Morphix.TTS")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] [TTS]: %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

# ---------------------------------------------------------------------------
# Voice constants (Official Microsoft Edge Neural Voices)
# ---------------------------------------------------------------------------
VOICE_EN_HINGLISH: str = "en-IN-NeerjaNeural"   # English & Hinglish (Female)
VOICE_HI_DEVANAGARI: str = "hi-IN-SwaraNeural"  # Devanagari Hindi (Female)
TTS_RATE: str = "+5%"                            # Lively, conversational pace
TTS_PITCH: str = "+2Hz"                          # Warm, cheerful tone


class TTSDispatcher:
    """
    Microsoft Edge TTS dispatcher engine for Morphix.

    Detects whether the input text is Hindi (Devanagari script) or
    English/Hinglish, selects the appropriate Aarti Neural voice,
    synthesizes speech to a temporary MP3 file, plays it via SoundPlayer,
    and auto-deletes the file afterwards.
    """

    def __init__(self, sound_player: Optional[Any] = None) -> None:
        """
        Initializes the TTSDispatcher and prepares the temp audio directory.

        :param sound_player: Optional SoundPlayer instance. A new one is
                             created automatically if not provided.
        """
        self.project_root: Path = self._find_project_root()
        self.temp_audio_dir: Path = self.project_root / "temp" / "audio"
        self.temp_audio_dir.mkdir(parents=True, exist_ok=True)

        self.sound_player: Any = sound_player if sound_player else SoundPlayer()
        self._is_speaking: bool = False

        self._cleanup_leftover_audio()
        self._log_startup_info()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_project_root(self) -> Path:
        """
        Resolves the project root directory by walking up the filesystem
        until a directory containing 'src' or 'config.py' is found.

        :return: Path to the project root.
        """
        current_file = Path(__file__).resolve()
        for parent in current_file.parents:
            if (parent / "src").exists() or (parent / "config.py").exists():
                return parent
        return current_file.parents[2]

    def _is_devanagari(self, text: str) -> bool:
        """
        Returns True if the text contains Hindi Devanagari characters
        (Unicode block U+0900–U+097F).

        :param text: Input text string.
        :return: True if Devanagari, False otherwise.
        """
        return any("\u0900" <= ch <= "\u097F" for ch in text)

    def _log_startup_info(self) -> None:
        """Logs Edge TTS configuration at application startup."""
        logger.info("==================================================")
        logger.info("        MORPHIX TTS — MICROSOFT EDGE TTS          ")
        logger.info("==================================================")
        logger.info("Engine           : edge-tts (Microsoft Edge TTS)")
        logger.info("English Voice    : %s", VOICE_EN_HINGLISH)
        logger.info("Hindi Voice      : %s", VOICE_HI_DEVANAGARI)
        logger.info("Rate             : %s", TTS_RATE)
        logger.info("Pitch            : %s", TTS_PITCH)
        logger.info("Temp Audio Dir   : %s", self.temp_audio_dir)
        logger.info("==================================================")

    def _cleanup_leftover_audio(self) -> None:
        """
        Removes all leftover .mp3 files inside temp_audio_dir on startup
        to prevent storage bloat from previous interrupted sessions.
        """
        try:
            if self.temp_audio_dir.exists():
                for item in self.temp_audio_dir.iterdir():
                    if item.is_file() and item.suffix.lower() in (".mp3", ".wav"):
                        try:
                            os.remove(str(item))
                            logger.info(
                                "Removed leftover audio file on startup: %s", item.name
                            )
                        except Exception as exc:
                            logger.warning(
                                "Failed to remove leftover audio file %s: %s",
                                item.name, str(exc)
                            )
        except Exception as exc:
            logger.error("Error during leftover audio startup cleanup: %s", str(exc))

    async def _synthesize_async(self, text: str, voice: str, output_path: Path) -> bool:
        """
        Async coroutine that synthesizes speech with edge_tts.Communicate
        and streams the result to an MP3 file. Includes automatic fallback
        without rate/pitch formatting if synthesis fails.

        :param text: Text to synthesize.
        :param voice: Edge TTS voice name.
        :param output_path: Destination MP3 file path.
        :return: True on success, False on failure.
        """
        try:
            communicate = edge_tts.Communicate(
                text=text,
                voice=voice,
                rate=TTS_RATE,
                pitch=TTS_PITCH
            )
            await communicate.save(str(output_path))
            if output_path.exists() and output_path.stat().st_size > 0:
                return True
        except Exception as exc:
            logger.warning("Edge TTS custom parameter synthesis failed (%s), attempting default parameters fallback...", exc)

        # Fallback 1: Default pitch & rate
        try:
            communicate = edge_tts.Communicate(text=text, voice=voice)
            await communicate.save(str(output_path))
            if output_path.exists() and output_path.stat().st_size > 0:
                return True
        except Exception as exc:
            logger.error("Edge TTS synthesis failed: %s", str(exc))

        # Fallback 2: General Indian voice fallback if target voice fails
        fallback_voice = VOICE_EN_HINGLISH if voice != VOICE_EN_HINGLISH else "en-IN-NeerjaExpressiveNeural"
        try:
            communicate = edge_tts.Communicate(text=text, voice=fallback_voice)
            await communicate.save(str(output_path))
            return output_path.exists() and output_path.stat().st_size > 0
        except Exception as exc:
            logger.error("Edge TTS fallback synthesis failed: %s", str(exc))
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(self, text: str, *args, **kwargs) -> bool:
        """
        Synthesizes speech from text using Microsoft Edge TTS (Aarti Neural)
        and plays it through the system speaker via SoundPlayer.

        Language routing:
          - Devanagari text → hi-IN-AartiNeural
          - English/Hinglish → en-IN-AartiNeural

        Saves a temporary MP3 to temp/audio/speech_<timestamp_ms>.mp3,
        plays it blocking, then deletes it immediately.

        :param text: Text string to speak.
        :return: True if speech was synthesized and played successfully,
                 False otherwise.
        """
        if not text or not text.strip():
            logger.warning("Cannot speak empty or whitespace-only text.")
            return False

        # Select voice based on script detection
        if self._is_devanagari(text):
            voice = VOICE_HI_DEVANAGARI
            lang_tag = "Hindi (hi-IN-AartiNeural)"
        else:
            voice = VOICE_EN_HINGLISH
            lang_tag = "English/Hinglish (en-IN-AartiNeural)"

        logger.info("Routing to Edge TTS [%s]", lang_tag)

        # Build temp MP3 path
        timestamp_ms = int(time.time() * 1000)
        temp_mp3 = self.temp_audio_dir / f"speech_{timestamp_ms}.mp3"

        self._is_speaking = True
        # Audio ducking: lower music volume while speaking
        try:
            from src.intent.music_plugin import MusicService
            MusicService.get_instance().duck_for_tts()
        except Exception:
            pass

        try:
            # Run async synthesis in a clean event loop
            success = asyncio.run(self._synthesize_async(text, voice, temp_mp3))

            if not success:
                logger.error("Edge TTS synthesis produced no audio output.")
                return False

            logger.info("Playing speech via SoundPlayer [%s]...", lang_tag)
            self.sound_player.play_sound(temp_mp3, blocking=True)
            logger.info("Finished speaking [%s].", lang_tag)
            return True

        except Exception as exc:
            logger.error("TTS speak() failed: %s", str(exc))
            return False

        finally:
            self._is_speaking = False
            # Restore music volume after speaking
            try:
                from src.intent.music_plugin import MusicService
                MusicService.get_instance().restore_after_tts()
            except Exception:
                pass

            # Always attempt to clean up the temp file
            try:
                if temp_mp3.exists():
                    os.remove(str(temp_mp3))
                    logger.info("Deleted temporary audio file: %s", temp_mp3.name)
            except Exception as exc:
                logger.warning(
                    "Failed to remove temporary audio file %s: %s",
                    temp_mp3.name, str(exc)
                )

    def stop(self) -> None:
        """
        Signals that active TTS playback should cease.
        Delegates stop to SoundPlayer if supported.
        """
        logger.info("Stopping TTS playback...")
        self._is_speaking = False
        try:
            if hasattr(self.sound_player, "stop"):
                self.sound_player.stop()
        except Exception as exc:
            logger.error("Error stopping SoundPlayer: %s", str(exc))


class TextToSpeech(TTSDispatcher):
    """
    Microsoft Edge TTS engine for Morphix.
    Inherits all synthesis and playback logic from TTSDispatcher.

    Usage:
        tts = TextToSpeech()
        tts.speak("Hello Tanisha!")
    """
    pass

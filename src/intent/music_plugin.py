"""
Morphix Desktop Companion Robot - Free Music Player Plugin
===========================================================
File: src/intent/music_plugin.py
Description:
    Zero-subscription YouTube direct audio streaming using yt-dlp,
    with hardware-direct audio streaming (ffmpeg PCM pipe to sounddevice /
    ffplay / mpv / pygame.mixer fallback), smart query extraction,
    and bilingual voice control.
"""

from dataclasses import dataclass
import logging
import os
import platform
import re
import shutil
import subprocess
import threading
import time
from typing import Any, Callable, Dict, Optional
import numpy as np

logger = logging.getLogger("Music")

try:
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG_EXE = shutil.which("ffmpeg")


def clean_leading_fillers(text: str) -> str:
    """Recursively removes conversational and wake filler prefixes."""
    cleaned = text.strip()
    while True:
        prev = cleaned
        cleaned = re.sub(
            r'^(?:ok|okay|hey|bro|morphix|morfix|hello|hi|yo|can you|could you|would you|will you|please|plz|kripya|zara|ek baar)\s+',
            '', cleaned, flags=re.IGNORECASE
        ).strip()
        if cleaned == prev:
            break
    return cleaned


@dataclass
class TrackInfo:
    title: str
    url: str
    duration: int = 0
    channel: str = ""
    is_playing: bool = False
    is_paused: bool = False


class MusicService:
    """Manages audio stream resolution via yt-dlp and direct sound-card hardware streaming."""

    _instance: Optional["MusicService"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.current_track: Optional[TrackInfo] = None
        self._is_loading = False
        self._volume = 0.7  # Default 70%
        self._proc: Optional[subprocess.Popen] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._stream_thread: Optional[threading.Thread] = None
        self._playback_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "MusicService":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def set_volume(self, volume: float) -> None:
        """Sets internal digital playback volume (0.0 to 1.0)."""
        self._volume = max(0.0, min(1.0, float(volume)))

    def search_and_get_stream(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Uses yt-dlp to search for the best audio stream URL and metadata.
        Returns the raw streaming URL without downloading full video/audio files.
        """
        try:
            import yt_dlp

            ydl_opts = {
                "format": "bestaudio/best",
                "quiet": True,
                "no_warnings": True,
                "extract_flat": False,
                "default_search": "ytsearch1",
                "noplaylist": True,
                "socket_timeout": 8,
            }

            search_query = query.strip()
            if not any(k in search_query.lower() for k in ("audio", "song", "music", "track", "remix", "lofi")):
                search_query = f"{search_query} audio"

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                logger.info("Searching YouTube for query: '%s'", search_query)
                info = ydl.extract_info(f"ytsearch1:{search_query}", download=False)
                if not info or "entries" not in info or not info["entries"]:
                    logger.warning("No YouTube search results for: '%s'", search_query)
                    return None

                entry = info["entries"][0]
                audio_url = entry.get("url")
                title = entry.get("title", query)
                duration = entry.get("duration", 0)
                channel = entry.get("uploader", "")

                return {
                    "title": title,
                    "url": audio_url,
                    "duration": duration,
                    "channel": channel,
                }
        except Exception as ex:
            logger.error("yt-dlp search failed for '%s': %s", query, ex)
            return None

    def play(self, query: str, on_started: Optional[Callable[[TrackInfo], None]] = None) -> Dict[str, Any]:
        """
        Asynchronously searches and starts direct hardware streaming for the requested song query.
        """
        def _play_worker():
            self._is_loading = True
            try:
                res = self.search_and_get_stream(query)
                if not res or not res.get("url"):
                    logger.warning("Could not resolve audio stream for: %s", query)
                    self._is_loading = False
                    return

                title = res["title"]
                stream_url = res["url"]

                with self._playback_lock:
                    self._kill_current_stream()

                    track = TrackInfo(
                        title=title,
                        url=stream_url,
                        duration=res.get("duration", 0),
                        channel=res.get("channel", ""),
                        is_playing=True,
                        is_paused=False
                    )
                    self.current_track = track
                    self._stop_event.clear()
                    self._pause_event.clear()

                if on_started:
                    on_started(track)

                # Start the background audio streaming loop
                self._stream_audio_loop(stream_url)

            except Exception as play_ex:
                logger.error("Audio playback error: %s", play_ex)
            finally:
                self._is_loading = False

        thread = threading.Thread(target=_play_worker, daemon=True, name="MusicPlayerThread")
        thread.start()

        # Clean query for speech confirmation
        clean_title = re.sub(r'^(?:play\s+some|play\s+song|play|chalao|bajao)\s*', '', query, flags=re.IGNORECASE).strip().title()
        if not clean_title:
            clean_title = query.title()

        return {
            "status": "searching",
            "query": query,
            "title": clean_title,
        }

    def _stream_audio_loop(self, stream_url: str) -> None:
        """Streams PCM audio directly into sounddevice hardware output using ffmpeg."""
        import sounddevice as sd

        if not FFMPEG_EXE:
            logger.error("FFMPEG binary not found — cannot stream audio.")
            return

        cmd = [
            FFMPEG_EXE,
            "-reconnect", "1",
            "-reconnect_streamed", "1",
            "-reconnect_delay_max", "5",
            "-i", stream_url,
            "-f", "s16le",
            "-ar", "44100",
            "-ac", "2",
            "-vn",
            "-loglevel", "quiet",
            "-"
        ]

        stream_out = None
        try:
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=65536)
            stream_out = sd.RawOutputStream(samplerate=44100, channels=2, dtype='int16')
            stream_out.start()
            logger.info("Direct soundcard audio stream running.")

            while not self._stop_event.is_set():
                if self._pause_event.is_set():
                    time.sleep(0.1)
                    continue

                chunk = self._proc.stdout.read(4096)
                if not chunk:
                    break

                if self._volume != 1.0:
                    arr = np.frombuffer(chunk, dtype=np.int16)
                    arr = (arr * self._volume).astype(np.int16)
                    stream_out.write(arr.tobytes())
                else:
                    stream_out.write(chunk)

        except Exception as ex:
            logger.warning("Streaming loop exception: %s", ex)
        finally:
            if stream_out:
                try:
                    stream_out.stop()
                    stream_out.close()
                except Exception:
                    pass
            self._kill_current_stream()
            if self.current_track:
                self.current_track.is_playing = False
            logger.info("Direct soundcard audio stream finished.")

    def _kill_current_stream(self) -> None:
        """Kills any active ffmpeg streaming subprocess."""
        self._stop_event.set()
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=0.5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None

    def pause(self) -> str:
        """Pauses the currently playing track."""
        if self.is_playing():
            self._pause_event.set()
            if self.current_track:
                self.current_track.is_paused = True
                self.current_track.is_playing = False
            logger.info("Music playback paused.")
            return "Music paused."
        return "No music is currently playing."

    def resume(self) -> str:
        """Resumes the paused track."""
        if self._pause_event.is_set():
            self._pause_event.clear()
            if self.current_track:
                self.current_track.is_paused = False
                self.current_track.is_playing = True
            logger.info("Music playback resumed.")
            return "Music resumed."
        elif self.current_track and self.current_track.title:
            return self.play(self.current_track.title).get("title", "Music")
        return "Could not resume music."

    def stop(self) -> str:
        """Stops music playback completely."""
        with self._playback_lock:
            self._kill_current_stream()
            if self.current_track:
                self.current_track.is_playing = False
                self.current_track.is_paused = False
                self.current_track = None
            logger.info("Music playback stopped.")
            return "Music stopped."

    def is_playing(self) -> bool:
        """Returns True if music is currently streaming."""
        return (
            self.current_track is not None and
            self.current_track.is_playing and
            not self._pause_event.is_set() and
            self._proc is not None and
            self._proc.poll() is None
        )

    def get_now_playing(self) -> Optional[TrackInfo]:
        """Returns the current track information."""
        return self.current_track

    def format_speech_response(self, action: str, title: str = "", is_hindi: bool = False) -> str:
        """Generates natural bilingual speech confirmation."""
        if action == "play":
            if is_hindi:
                return f"{title} play kar rahi hoon, enjoy kijiye!"
            return f"Playing {title} for you."
        elif action == "pause":
            if is_hindi:
                return "Gaana pause kar diya."
            return "Music paused."
        elif action == "resume":
            if is_hindi:
                return "Gaana resume kar diya."
            return "Resuming music."
        elif action == "stop":
            if is_hindi:
                return "Music band kar diya."
            return "Music stopped."
        elif action == "next":
            if is_hindi:
                return "Agla gaana chala rahi hoon."
            return "Playing the next track."
        return "Done."


class MusicIntentPlugin:
    """NLU plugin for detecting music playback commands with smart query extraction."""

    def __init__(self, service: Optional[MusicService] = None) -> None:
        self.service = service or MusicService.get_instance()

    def match(self, command: str) -> Optional[Any]:
        """
        Classifies whether the user command is a music control or playback request.
        Returns a classified IntentResult or None.
        """
        from src.intent.intent_classifier import IntentResult, IntentType

        c = clean_leading_fillers(command.strip())
        is_hindi = bool(re.search(r'\b(gaana|karo|baja|bajao|chalao|suno|sunao|band|roko)\b', c, flags=re.IGNORECASE))

        # 1. Stop Music Commands
        if re.search(r'\b(stop music|stop the music|stop song|stop the song|stop playing|band karo|gaana band|music band|chup karo|stop)\b', c, flags=re.IGNORECASE):
            return IntentResult(
                intent=IntentType.STOP_MUSIC,
                confidence=0.95,
                parameters={"action": "stop", "is_hindi": is_hindi}
            )

        # 2. Pause Music Commands
        if re.search(r'\b(pause music|pause the music|pause song|pause the song|pause karo|gaana roko|pause)\b', c, flags=re.IGNORECASE):
            return IntentResult(
                intent=IntentType.PAUSE_MUSIC,
                confidence=0.95,
                parameters={"action": "pause", "is_hindi": is_hindi}
            )

        # 3. Resume Music Commands
        if re.search(r'\b(resume music|resume the music|resume song|resume the song|unpause|continue song|continue music|gaana continue|chalu karo|resume)\b', c, flags=re.IGNORECASE):
            return IntentResult(
                intent=IntentType.RESUME_MUSIC,
                confidence=0.95,
                parameters={"action": "resume", "is_hindi": is_hindi}
            )

        # 4. Next Track Commands
        if re.search(r'\b(next song|next track|play next|agla gaana|change song|skip song|next)\b', c, flags=re.IGNORECASE):
            return IntentResult(
                intent=IntentType.NEXT_MUSIC,
                confidence=0.95,
                parameters={"action": "next", "is_hindi": is_hindi}
            )

        # 5. Play Music Patterns
        patterns = [
            r'^(?:play\s+some|play\s+song|play|chalao|bajao|listen\s+to|put\s+on|gaana\s+bajao|gaana\s+chalao|music\s+chalao|start\s+playing)\s+(.*)',
            r'^(?:song|gaana)\s+(?:play\s+karo|chalao|bajao)\s+(.*)',
            r'(.*)\s+(?:play\s+karo|chala\s+do|baja\s+do|chalao|bajao)$',
        ]

        query = None
        for p in patterns:
            m = re.search(p, c, flags=re.IGNORECASE)
            if m:
                query = m.group(1).strip()
                break

        if not query and c.lower() in ("play", "play music", "play some music", "gaana bajao", "gaana chalao", "start music", "play a song", "kuch gaana bajao"):
            query = "top hits trending"

        if query:
            # Clean trailing politeness / conversational filler words
            query = re.sub(r'\b(for\s+me|please|plz|yaar|bhi)\b', '', query, flags=re.IGNORECASE).strip()
            if query.lower() in ("music", "song", "gaana", "kuch", "some music", "a song"):
                query = "top hits trending"
            elif len(query.split()) > 1 and query.lower().endswith((" song", " gaana")):
                query = re.sub(r'\s+(?:song|gaana)$', '', query, flags=re.IGNORECASE).strip()
            query = re.sub(r'^some\s+', '', query, flags=re.IGNORECASE).strip()

            return IntentResult(
                intent=IntentType.PLAY_MUSIC,
                confidence=0.95,
                parameters={"action": "play", "query": query, "is_hindi": is_hindi}
            )

        return None

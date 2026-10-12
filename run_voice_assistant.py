from __future__ import annotations

import collections
from enum import Enum
import io
import logging
from pathlib import Path
import queue
import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
import time
import wave
from typing import Any, Optional, Tuple
import speech_recognition as sr


# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Placeholders for modules imported dynamically inside main() after environment loading
Config = None
SoundPlayer = None
IntentClassifier = None
IntentType = None
ReminderChecker = None
ReminderManager = None
TodoManager = None
TranscriptCleaner = None
ReminderIntentParser = None
TodoParser = None
AIRegistry = None
SpeechToText = None
TextToSpeech = None
VoiceAssistant = None
WakeWordDetector = None
OpenWakeWordEngine = None
WakeWordManager = None
groq_chat_mgr = None
eye_display = None   # EyeAnimatorThread instance (daemon, started in main())

# Setup application logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [VoiceAssistantApp]: %(message)s"
)
logger = logging.getLogger("VoiceAssistantApp")


class AssistantState(Enum):
    INITIALIZING = "INITIALIZING"
    WAIT_WAKE = "WAIT_WAKE"
    ACTIVE_MODE = "ACTIVE_MODE"
    COMMAND_RECEIVED = "COMMAND_RECEIVED"
    SLEEP = "SLEEP"


_current_state = AssistantState.INITIALIZING


def transition_state(new_state: AssistantState) -> None:
    global _current_state
    old_state = _current_state
    _current_state = new_state
    logger.info("Transitioning state: %s -> %s", old_state.name, new_state.name)
    print(f"{old_state.name}\n↓\n{new_state.name}")


def is_meaningful_speech(text: str) -> bool:
    """
    Validates if the recognized speech transcript contains meaningful commands.
    Returns False for blank transcripts, [BLANK_AUDIO], microphone noise, empty strings, etc.
    """
    if not text:
        return False
    cleaned = text.strip()
    if not cleaned:
        return False
    if cleaned.upper() == "[BLANK_AUDIO]":
        return False
    if cleaned.lower() in ["[noise]", "noise", "[microphone noise]"]:
        return False
    return True


_last_stt_offline_warning_time: float = 0.0


def transcribe_with_google(wav_bytes: bytes) -> str:
    """
    Transcribes raw WAV bytes using Google Web Speech API (speech_recognition).
    Attempts primary en-IN transcription, falling back to hi-IN on low confidence.
    Gracefully catches DNS/socket/network dropouts, throttles terminal error spam,
    and transitions the Pygame display to the NO_INTERNET alert overlay.
    """
    global _last_stt_offline_warning_time
    import io
    import logging
    import time
    import speech_recognition as sr
    from src.display.eye_animator import get_active_display

    logger = logging.getLogger("VoiceAssistantApp.GoogleSTT")
    recognizer = sr.Recognizer()
    try:
        wav_io = io.BytesIO(wav_bytes)
        with sr.AudioFile(wav_io) as source:
            audio_data = recognizer.record(source)

        # Pass 1: Try en-IN primary recognition
        try:
            result = recognizer.recognize_google(audio_data, language="en-IN", show_all=True)
            if isinstance(result, dict) and 'alternative' in result and len(result['alternative']) > 0:
                best_alt = result['alternative'][0]
                transcript = best_alt.get('transcript', '')
                confidence = best_alt.get('confidence', 1.0)

                # Fallback: if confidence is low, re-transcribe using hi-IN
                if confidence < 0.65:
                    logger.info("Google STT primary en-IN confidence %.2f below threshold (0.65). Falling back to hi-IN...", confidence)
                    try:
                        fallback_transcript = recognizer.recognize_google(audio_data, language="hi-IN")
                        if fallback_transcript:
                            return fallback_transcript
                    except Exception:
                        pass
                return transcript
            else:
                # If no alternatives returned, fallback directly
                return recognizer.recognize_google(audio_data, language="hi-IN")
        except sr.UnknownValueError:
            # If primary recognition fails, attempt fallback to hi-IN
            return recognizer.recognize_google(audio_data, language="hi-IN")
        except Exception as e:
            # Check if this is a network / DNS error
            err_str = str(e).lower()
            is_net_err = isinstance(e, (sr.RequestError, OSError, ConnectionError)) or any(
                k in err_str for k in ["11001", "getaddrinfo", "nodename", "connection", "unreachable", "name or service not known", "timed out"]
            )
            if is_net_err:
                raise e
            logger.warning("Google STT primary en-IN recognition failed, attempting fallback to hi-IN: %s", str(e))
            return recognizer.recognize_google(audio_data, language="hi-IN")

    except sr.UnknownValueError:
        # Graceful handling: return empty text, silence hallucinations
        return ""
    except Exception as e:
        err_str = str(e).lower()
        is_net_err = isinstance(e, (sr.RequestError, OSError, ConnectionError)) or any(
            k in err_str for k in ["11001", "getaddrinfo", "nodename", "connection", "unreachable", "name or service not known", "timed out", "request error"]
        )
        if is_net_err:
            try:
                disp = get_active_display()
                if disp:
                    disp.set_overlay("NO_INTERNET", timeout=5.0)
            except Exception:
                pass

            now = time.time()
            if now - _last_stt_offline_warning_time >= 5.0:
                _last_stt_offline_warning_time = now
                logger.warning("[Morphix] Network offline / DNS error during STT: %s", str(e))
        else:
            logger.error("Error during Google Speech Recognition: %s", e)
        return ""


class ContinuousMicrophoneStream:
    """
    Continuous microphone stream with a 500ms audio pre-roll ring buffer.
    Prevents initial consonant truncation (e.g. 'H' in 'Hey Morphix') by attaching
    pre-speech audio frames to live speech recordings.
    """

    def __init__(self, sample_rate: int = 16000, blocksize: int = 800, preroll_ms: int = 500, sound_player: Optional[Any] = None) -> None:
        """
        Initializes continuous microphone stream and ring buffer.

        :param sample_rate: Audio sampling rate in Hz (default: 16000 Hz).
        :param blocksize: Audio buffer block size in samples (default: 800 = 50ms chunk).
        :param preroll_ms: Duration of pre-speech audio ring buffer in milliseconds (default: 500 ms).
        :param sound_player: Optional SoundPlayer instance to check if bark is playing.
        """
        self.sample_rate: int = sample_rate
        self.channels: int = 1
        self.blocksize: int = blocksize
        self.preroll_ms: int = preroll_ms
        self.sound_player: Optional[Any] = sound_player

        # Large ring buffer (4 seconds) to support sliding window wake-word detection
        max_blocks = int((4000 / 1000.0) * (sample_rate / blocksize))
        self.ring_buffer: collections.deque = collections.deque(maxlen=max_blocks)

        self._stream = None
        self._running: bool = False
        self.recording_queue: Optional[queue.Queue] = None

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        """Internal sounddevice stream callback."""
        if status:
            logger.debug("Audio stream callback status: %s", status)
        
        # Check if the bark sound is playing to discard input and prevent hearing itself
        is_bark_playing = False
        if self.sound_player:
            is_bark_playing = self.sound_player.is_playing()
        else:
            try:
                import pygame
                if pygame.mixer.get_init() and pygame.mixer.get_busy():
                    is_bark_playing = True
            except Exception:
                pass

        if is_bark_playing:
            # Overwrite with silent bytes to preserve timing
            data = b"\x00" * len(indata.tobytes())
        else:
            data = indata.tobytes()

        self.ring_buffer.append(data)
        if self.recording_queue is not None:
            self.recording_queue.put(data)

    def start(self) -> bool:
        """Starts continuous microphone capture stream."""
        if not self._running:
            try:
                import sounddevice as sd
                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype="int16",
                    blocksize=self.blocksize,
                    callback=self._audio_callback
                )
                self._stream.start()
                self._running = True
                logger.info("Continuous Microphone Stream started successfully.")
                logger.info("Sample Rate: %d Hz | Channels: %d (Mono) | Chunk Size: %d samples", self.sample_rate, self.channels, self.blocksize)
                return True
            except Exception as e:
                logger.error("Failed to start continuous microphone stream: %s", str(e))
                return False
        return True

    def clear(self) -> None:
        """Clears the audio ring buffer."""
        self.ring_buffer.clear()

    def get_last_n_seconds(self, seconds: float = 3.0) -> bytes:
        """
        Extracts the latest `seconds` of audio from the ring buffer and returns it as WAV bytes.
        """
        # Determine how many bytes represent `seconds`
        # 16000 samples/sec * 2 bytes/sample = 32000 bytes/sec
        bytes_needed = int(seconds * self.sample_rate * 2)
        
        # Get all raw PCM blocks currently in the ring buffer
        all_blocks = list(self.ring_buffer)
        raw_pcm = b"".join(all_blocks)
        
        # Take the tail of the buffer up to bytes_needed
        if len(raw_pcm) > bytes_needed:
            raw_pcm = raw_pcm[-bytes_needed:]
            
        # Wrap in WAV format
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(raw_pcm)
            
        return wav_buffer.getvalue()

    def record_chunk_with_preroll(self, duration: float = 3.0) -> Tuple[bytes, float, float]:
        """
        Retrieves pre-speech buffer (500ms) combined with live recorded audio frames.
        """
        speech_start_time = time.time()
        
        debug_pipeline = getattr(Config, "DEBUG_AUDIO_PIPELINE", True)
        if debug_pipeline:
            logger.info("[DIAGNOSTIC] Recording start timestamp: %.3f (%s)", speech_start_time, time.strftime('%H:%M:%S', time.localtime(speech_start_time)))

        # Slice only the requested pre-roll duration (500ms) from the ring buffer
        preroll_bytes = int((self.preroll_ms / 1000.0) * self.sample_rate * 2)
        all_preroll = b"".join(list(self.ring_buffer))
        preroll_pcm = all_preroll[-preroll_bytes:] if len(all_preroll) > preroll_bytes else all_preroll

        self.recording_queue = queue.Queue()

        # Warm up the queue to avoid immediate empty queue reads on startup
        warmup_timeout = getattr(Config, "COMMAND_QUEUE_WARMUP_TIMEOUT", 1.0)
        warmup_start = time.time()
        while self.recording_queue.empty() and (time.time() - warmup_start < warmup_timeout):
            time.sleep(0.01)
        
        warmup_duration = time.time() - warmup_start
        if debug_pipeline:
            logger.info("[DIAGNOSTIC] Recording queue warmup time: %.4f seconds", warmup_duration)

        num_blocks = int((duration * self.sample_rate) / self.blocksize)
        live_blocks = []
        first_frame_time = 0.0

        for _ in range(num_blocks):
            try:
                block = self.recording_queue.get(timeout=0.5)
                live_blocks.append(block)
                if not first_frame_time:
                    first_frame_time = time.time()
                    if debug_pipeline:
                        logger.info("[DIAGNOSTIC] First audio frame timestamp: %.3f (%s)", first_frame_time, time.strftime('%H:%M:%S', time.localtime(first_frame_time)))
            except queue.Empty:
                if self._running:
                    logger.debug("Audio queue transiently empty (expected in real-time streaming).")
                else:
                    logger.warning("Audio queue underflow in record_chunk_with_preroll (recording stream stopped).")

        self.recording_queue = None
        speech_end_time = time.time()
        if debug_pipeline:
            logger.info("[DIAGNOSTIC] Recording stop timestamp: %.3f (%s)", speech_end_time, time.strftime('%H:%M:%S', time.localtime(speech_end_time)))
            logger.info("[DIAGNOSTIC] Speech duration: %.2f seconds", speech_end_time - speech_start_time)

        full_pcm = preroll_pcm + b"".join(live_blocks)

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(full_pcm)

        return wav_buffer.getvalue(), speech_start_time, speech_end_time

    def stop(self) -> None:
        """Stops continuous microphone stream."""
        if self._running and self._stream:
            self._running = False
            try:
                self._stream.stop()
                self._stream.close()
                logger.info("Continuous Microphone Stream stopped.")
            except Exception as e:
                logger.error("Error stopping microphone stream: %s", str(e))


def strip_wake_prefix(text: str) -> str:
    """
    Strips common wake phrase prefixes from the beginning of the text
    to make rule-based parsing matching more robust.
    """
    lowercased = text.lower().strip()
    prefixes = [
        "morphix ", "hey morphix ", "hello morphix ",
        "hi morphix ", "wake up morphix ", "hey morfix ",
        "hello morfix ", "morphics ", "hey morphics ",
        "he might fix ", "hey more fix ", "hey more pics ",
        "more fix "
    ]
    for prefix in prefixes:
        if lowercased.startswith(prefix):
            return text[len(prefix):].strip()
    return text.strip()


def record_command_with_vad(
    mic_stream: ContinuousMicrophoneStream,
    stt: SpeechToText,
    max_duration: float = 10.0,
    silence_timeout: float = 1.5,
    post_tts_pause_ms: int = 50,
    active_timeout: float = 300.0,
    last_speech_time_ref: float = 0.0
) -> Tuple[Optional[bytes], str, bool, float]:
    """
    Returns a 4-tuple: (wav_bytes, transcript, timeout_reached, updated_last_speech_time).
    The 4th element lets callers update their inactivity reference without mutation bugs.
    """
    # (docstring moved above signature — see 4-tuple comment above)
    global eye_display
    import numpy as np

    recording_start_time = time.time()
    debug_pipeline = getattr(Config, "DEBUG_AUDIO_PIPELINE", True)
    if debug_pipeline:
        logger.info("[DIAGNOSTIC] Recording start timestamp: %.3f (%s)", recording_start_time, time.strftime('%H:%M:%S', time.localtime(recording_start_time)))

    # Initialize queue BEFORE the post-TTS pause to capture everything!
    mic_stream.recording_queue = queue.Queue()

    # Pause to allow audio echo/hardware transition to settle
    time.sleep(post_tts_pause_ms / 1000.0)

    # Warm up the queue to avoid reading from an empty queue
    warmup_timeout = getattr(Config, "COMMAND_QUEUE_WARMUP_TIMEOUT", 1.0)
    warmup_start = time.time()
    while mic_stream.recording_queue.empty() and (time.time() - warmup_start < warmup_timeout):
        time.sleep(0.01)

    warmup_duration = time.time() - warmup_start
    if debug_pipeline:
        logger.info("[DIAGNOSTIC] Recording queue warmup time: %.4f seconds", warmup_duration)

    logger.info("Command started (Waiting for speech...)")
    if debug_pipeline:
        print("Waiting for speech activity...")

    start_time = time.time()
    last_speech_time = time.time()
    speech_started = False
    last_log_time = 0.0
    first_frame_time = 0.0
    vad_trigger_time = 0.0

    accumulated_blocks = []
    # 500ms rolling pre-speech buffer so initial syllables ("Yes", "Hey") are never clipped before VAD trigger
    max_pre_blocks = max(1, int((500 / 1000.0) * (mic_stream.sample_rate / mic_stream.blocksize)))
    pre_speech_blocks = collections.deque(maxlen=max_pre_blocks)
    silence_duration = 0.0
    rms_list = []

    try:
        while True:
            # Check timeout if speech has not started yet
            if not speech_started:
                from src.intent.music_plugin import MusicService
                is_music_active = False
                try:
                    from src.intent.music_plugin import MusicService
                    is_music_active = MusicService.get_instance().is_playing()
                except Exception:
                    pass

                is_surveillance_active = False
                try:
                    from src.vision.camera_manager import CameraManager
                    is_surveillance_active = CameraManager.get_instance().is_surveillance_active()
                except Exception:
                    pass

                if is_surveillance_active or is_music_active:
                    # While surveillance guard mode is active or music is streaming, freeze the 15-second sleep timer completely
                    last_speech_time_ref = time.time()
                    if is_surveillance_active:
                        if eye_display is not None and eye_display.is_alive():
                            try:
                                from src.display.eye_animator import EyeState as _ES
                                if eye_display.get_state() != _ES.SENTRY_SCAN:
                                    eye_display.set_state(_ES.SENTRY_SCAN)
                            except Exception:
                                pass
                    elif is_music_active:
                        if eye_display is not None and eye_display.is_alive():
                            try:
                                from src.display.eye_animator import EyeState as _ES
                                if eye_display.get_state() != _ES.MUSIC_MODE:
                                    track = MusicService.get_instance().get_now_playing()
                                    title = track.title if track else "Music"
                                    eye_display.set_music_data(title, is_playing=True)
                                    eye_display.set_state(_ES.MUSIC_MODE)
                            except Exception:
                                pass
                else:
                    current_time = time.time()
                    elapsed_since_speech = current_time - last_speech_time_ref
                    remaining_time = max(0.0, active_timeout - elapsed_since_speech)

                    # Inactivity: At 11.0s of continuous silence (elapsed >= 11.0s or remaining <= 4.0s), transition to SLEEPY
                    if eye_display is not None and eye_display.is_alive():
                        try:
                            from src.display.eye_animator import EyeState as _ES
                            if (elapsed_since_speech >= 11.0 or remaining_time <= 4.0) and eye_display.get_state() != _ES.SLEEPY:
                                eye_display.set_state(_ES.SLEEPY)
                        except Exception:
                            pass

                    # Log countdown status every 1.0 second
                    if current_time - last_log_time >= 1.0:
                        last_speech_timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_speech_time_ref))
                        logger.info("Last Speech Timestamp: %s", last_speech_timestamp)
                        logger.info("Inactivity Timer: %.2f seconds", elapsed_since_speech)
                        logger.info("Remaining Time: %.2f seconds", remaining_time)
                        last_log_time = current_time

                    if elapsed_since_speech >= active_timeout:
                        logger.info("Inactivity timeout (%.1fs) reached. Returning to Deep Sleep.", active_timeout)
                        return None, "", True, last_speech_time_ref

            try:
                block = mic_stream.recording_queue.get(timeout=0.5)
                if not first_frame_time:
                    first_frame_time = time.time()
                    if debug_pipeline:
                        logger.info("[DIAGNOSTIC] First audio frame timestamp: %.3f (%s)", first_frame_time, time.strftime('%H:%M:%S', time.localtime(first_frame_time)))
            except queue.Empty:
                if mic_stream._running:
                    logger.debug("Audio queue transiently empty (expected in real-time streaming).")
                else:
                    logger.warning("Audio queue empty because stream is not running.")
                continue

            frame = np.frombuffer(block, dtype=np.int16)
            rms = np.sqrt(np.mean(frame.astype(np.float32) ** 2)) if len(frame) > 0 else 0.0
            rms_list.append(rms)

            if rms >= 100.0:
                if not speech_started:
                    speech_started = True
                    vad_trigger_time = time.time()
                    if debug_pipeline:
                        logger.info("[DIAGNOSTIC] VAD trigger timestamp: %.3f (%s)", vad_trigger_time, time.strftime('%H:%M:%S', time.localtime(vad_trigger_time)))
                    logger.info("Speech detected! Recording started.")
                    if debug_pipeline:
                        print("Recording...")
                    start_time = time.time()
                    # Prepend captured pre-speech blocks to eliminate start-of-speech clipping
                    accumulated_blocks.extend(pre_speech_blocks)
                    if eye_display is not None and eye_display.is_alive():
                        try:
                            from src.display.eye_animator import EyeState as _ES
                            eye_display.set_state(_ES.LISTENING)
                        except Exception:
                            pass
                
                last_speech_time = time.time()
                silence_duration = 0.0
            else:
                if not speech_started:
                    pre_speech_blocks.append(block)
                else:
                    silence_duration = time.time() - last_speech_time
                    if silence_duration >= silence_timeout:
                        logger.info("Silence detected (%.1fs >= %.1fs). Ending command capture.", silence_duration, silence_timeout)
                        break

            if speech_started:
                accumulated_blocks.append(block)

            # Avoid infinite loop or lockup
            if speech_started and (time.time() - start_time >= max_duration):
                logger.info("Command listening window reached maximum duration (%.1fs).", max_duration)
                break
    finally:
        mic_stream.recording_queue = None
        recording_stop_time = time.time()
        if debug_pipeline:
            logger.info("[DIAGNOSTIC] Recording stop timestamp: %.3f (%s)", recording_stop_time, time.strftime('%H:%M:%S', time.localtime(recording_stop_time)))
            speech_dur = (last_speech_time - vad_trigger_time) if speech_started else 0.0
            logger.info("[DIAGNOSTIC] Speech duration: %.2f seconds", speech_dur)
            logger.info("[DIAGNOSTIC] Silence duration: %.2f seconds", silence_duration)
            logger.info("[DIAGNOSTIC] Average RMS: %.2f", np.mean(rms_list) if rms_list else 0.0)
            logger.info("[DIAGNOSTIC] Peak RMS: %.2f", np.max(rms_list) if rms_list else 0.0)
        if debug_pipeline:
            print("Recording stopped")

    if not speech_started:
        return None, "", False, last_speech_time_ref

    logger.info("Transcribing recording...")

    # Prepend pre-roll ring buffer (sliced to preroll_ms)
    preroll_bytes = int((mic_stream.preroll_ms / 1000.0) * mic_stream.sample_rate * 2)
    all_preroll = b"".join(list(mic_stream.ring_buffer))
    preroll_pcm = all_preroll[-preroll_bytes:] if len(all_preroll) > preroll_bytes else all_preroll
    full_pcm = preroll_pcm + b"".join(accumulated_blocks)

    recording_duration = time.time() - start_time
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(mic_stream.channels)
        wf.setsampwidth(2)
        wf.setframerate(mic_stream.sample_rate)
        wf.writeframes(full_pcm)

    final_wav_bytes = wav_buffer.getvalue()
    # Always use Google STT — no local model required
    final_sentence = transcribe_with_google(final_wav_bytes)

    if debug_pipeline:
        print(f"Recording duration: {recording_duration:.2f}s")
        print(f"Silence duration: {silence_duration:.2f}s")

    logger.info("Final transcript: %s", final_sentence)
    logger.info("Recording Duration: %.2f seconds", recording_duration)
    logger.info("Silence Duration: %.2f seconds", silence_duration)

    # Return updated last_speech_time so callers can advance their inactivity reference
    return final_wav_bytes, final_sentence, False, time.time()


def handle_incoming_command(
    cleaned_command: str,
    raw_transcript: str,
    reminder_mgr: ReminderManager,
    todo_mgr: TodoManager,
    sound_player: SoundPlayer,
    wake_sound: str,
    tts: TextToSpeech,
    router: Optional[Any],
    ai_registry: AIRegistry,
    intent_classifier: Any
) -> None:
    """
    Orchestrates command parsing: Gemini NLU is prioritized first to understand intent
    semantically and correct STT/Whisper transcription errors. Fallbacks to local regex
    parsers or legacy plugins are executed if Gemini is unavailable or unknown.
    """
    global eye_display, groq_chat_mgr
    import time
    from src.gemini.gemini_logger import log_gemini_event

    # Strip wake phrase prefix (e.g. "Morphix remind me to...")
    cleaned_command = strip_wake_prefix(cleaned_command)

    # Language detection
    detected_lang = "English"
    if any('\u0900' <= char <= '\u097F' for char in cleaned_command):
        detected_lang = "Hindi"
    else:
        # Check Hinglish keywords
        hinglish_keywords = {
            "mujhe", "mera", "mere", "yaad", "dilana", "remind", "karna", "karo",
            "hata", "daal", "mein", "dikhao", "batao", "kya", "hai", "saare", "kar", "todo"
        }
        import re
        words = set(re.findall(r'\b\w+\b', cleaned_command.lower()))
        if words & hinglish_keywords:
            detected_lang = "Hinglish"

    print(f"[Morphix] 🌐 Detected Language: {detected_lang}")

    # 1. Prioritize Gemini NLU if active and available
    if getattr(Config, "GEMINI_ENABLED", False) and router and router.is_available():
        try:
            start_time = time.time()
            gemini_engine = ai_registry.get_provider("Gemini")
            validated_intent = gemini_engine.process_transcript(cleaned_command)
            intent = validated_intent.get("intent", "UNKNOWN")
            latency = time.time() - start_time
            
            if intent != "UNKNOWN":
                logger.info(
                    "[NLU LOG] Raw: '%s' | Cleaned: '%s' | Intent: '%s' | Source: GEMINI | Latency: %.3fs",
                    raw_transcript, cleaned_command, intent, latency
                )
                response = router.route(validated_intent)
                
                # Audio bark for todo/reminder additions/removals
                if intent in ["ADD_TODO", "DELETE_TODO", "EDIT_TODO", "COMPLETE_TODO", "SHOW_TODO", "SET_REMINDER", "DELETE_REMINDER", "SHOW_REMINDERS"]:
                    try:
                        sound_player.play_sound(wake_sound)
                    except Exception as ex:
                        logger.warning("Sound playback failed for Gemini route: %s", str(ex))
                
                print(f"[Morphix] 🤖 Response: {response}")
                if eye_display is not None and eye_display.is_alive():
                    try:
                        from src.display.eye_animator import EyeState as _ES
                        eye_display.set_state(_ES.SPEAKING)
                    except Exception:
                        pass
                try:
                    tts.speak(response)
                except Exception as ex:
                    logger.warning("TTS speaking failed for Gemini route: %s", str(ex))
                if eye_display is not None and eye_display.is_alive():
                    try:
                        from src.display.eye_animator import EyeState as _ES
                        eye_display.set_state(_ES.LISTENING)
                    except Exception:
                        pass
                return
        except Exception as ex:
            logger.error("Gemini Brain processing failed: %s. Falling back to local defaults.", str(ex))

    # 2. Step 1: Check Local Plugins (TodoIntentPlugin, ReminderIntentPlugin via intent_classifier)
    result = intent_classifier.classify(cleaned_command)
    
    if result and result.intent != IntentType.UNKNOWN and result.confidence >= 0.7:
        print("[Morphix] 🌐 Mode: LOCAL INTENT")
        log_gemini_event(
            request_type="intent",
            intent=result.intent.value,
            latency=0.0,
            success=True,
            model_name="local-classifier",
            source="LOCAL"
        )
        logger.info(
            "[NLU LOG] Raw: '%s' | Cleaned: '%s' | Intent: '%s' | Source: LOCAL | Latency: 0.000s",
            raw_transcript, cleaned_command, result.intent.value
        )
        
        # Audio bark for todo/reminder additions/removals/etc.
        if result.intent in [
            IntentType.ADD_TODO, IntentType.DELETE_TODO, IntentType.COMPLETE_TODO,
            IntentType.CLEAR_TODO, IntentType.ADD_REMINDER, IntentType.DELETE_REMINDER
        ]:
            try:
                sound_player.play_sound(wake_sound)
            except Exception as ex:
                logger.warning("Sound playback failed: %s", str(ex))

        # Check if Music, Volume, Telemetry, Weather, To-Do or Reminder intent
        is_hindi = (detected_lang in ("Hindi", "Hinglish")) or result.parameters.get("is_hindi", False)

        # 1. Music Intents
        if result.intent == IntentType.PLAY_MUSIC:
            from src.intent.music_plugin import MusicService
            ms = MusicService.get_instance()
            query = result.parameters.get("query", "popular trending music")
            title = query.title()
            response = ms.format_speech_response("play", title=title, is_hindi=is_hindi)
            print(f"[Morphix] 🎵 Music Query: {query}")
            print(f"[Morphix] 🤖 Response: {response}")

            def _on_music_started(track):
                if eye_display is not None and eye_display.is_alive():
                    try:
                        from src.display.eye_animator import EyeState as _ES
                        eye_display.set_music_data(track.title, is_playing=True)
                        eye_display.set_state(_ES.MUSIC_MODE)
                    except Exception:
                        pass

            # 1. Show SPEAKING eye and speak Aarti confirmation
            if eye_display is not None and eye_display.is_alive():
                try:
                    from src.display.eye_animator import EyeState as _ES
                    eye_display.set_music_data(title, is_playing=True)
                    eye_display.set_state(_ES.SPEAKING)
                except Exception:
                    pass
            try:
                tts.speak(response)
            except Exception as ex:
                logger.warning("TTS speaking failed for music: %s", str(ex))

            # 2. Trigger asynchronous background stream
            ms.play(query, on_started=_on_music_started)

            # 3. Transition to MUSIC_MODE visualizer
            if eye_display is not None and eye_display.is_alive():
                try:
                    from src.display.eye_animator import EyeState as _ES
                    eye_display.set_music_data(title, is_playing=True)
                    eye_display.set_state(_ES.MUSIC_MODE)
                except Exception:
                    pass
            return

        elif result.intent == IntentType.PAUSE_MUSIC:
            from src.intent.music_plugin import MusicService
            ms = MusicService.get_instance()
            ms.pause()
            response = ms.format_speech_response("pause", is_hindi=is_hindi)
            print(f"[Morphix] 🤖 Response: {response}")
            if eye_display is not None and eye_display.is_alive():
                try:
                    from src.display.eye_animator import EyeState as _ES
                    eye_display.set_state(_ES.SPEAKING)
                except Exception:
                    pass
            try:
                tts.speak(response)
            except Exception:
                pass
            if eye_display is not None and eye_display.is_alive():
                try:
                    from src.display.eye_animator import EyeState as _ES
                    eye_display.set_state(_ES.LISTENING)
                except Exception:
                    pass
            return

        elif result.intent == IntentType.RESUME_MUSIC:
            from src.intent.music_plugin import MusicService
            ms = MusicService.get_instance()
            ms.resume()
            response = ms.format_speech_response("resume", is_hindi=is_hindi)
            print(f"[Morphix] 🤖 Response: {response}")
            if eye_display is not None and eye_display.is_alive():
                try:
                    from src.display.eye_animator import EyeState as _ES
                    eye_display.set_state(_ES.SPEAKING)
                except Exception:
                    pass
            try:
                tts.speak(response)
            except Exception:
                pass
            if eye_display is not None and eye_display.is_alive():
                try:
                    from src.display.eye_animator import EyeState as _ES
                    track = ms.get_now_playing()
                    title = track.title if track else "Music"
                    eye_display.set_music_data(title, is_playing=True)
                    eye_display.set_state(_ES.MUSIC_MODE)
                except Exception:
                    pass
            return

        elif result.intent == IntentType.STOP_MUSIC:
            from src.intent.music_plugin import MusicService
            ms = MusicService.get_instance()
            ms.stop()
            response = ms.format_speech_response("stop", is_hindi=is_hindi)
            print(f"[Morphix] 🤖 Response: {response}")
            if eye_display is not None and eye_display.is_alive():
                try:
                    from src.display.eye_animator import EyeState as _ES
                    eye_display.set_state(_ES.SPEAKING)
                except Exception:
                    pass
            try:
                tts.speak(response)
            except Exception:
                pass
            if eye_display is not None and eye_display.is_alive():
                try:
                    from src.display.eye_animator import EyeState as _ES
                    eye_display.set_state(_ES.LISTENING)
                except Exception:
                    pass
            return

        elif result.intent == IntentType.NEXT_MUSIC:
            from src.intent.music_plugin import MusicService
            ms = MusicService.get_instance()
            response = ms.format_speech_response("next", is_hindi=is_hindi)
            print(f"[Morphix] 🤖 Response: {response}")
            if eye_display is not None and eye_display.is_alive():
                try:
                    from src.display.eye_animator import EyeState as _ES
                    eye_display.set_state(_ES.SPEAKING)
                except Exception:
                    pass
            try:
                tts.speak(response)
            except Exception:
                pass

            def _on_next_started(track):
                if eye_display is not None and eye_display.is_alive():
                    try:
                        from src.display.eye_animator import EyeState as _ES
                        eye_display.set_music_data(track.title, is_playing=True)
                        eye_display.set_state(_ES.MUSIC_MODE)
                    except Exception:
                        pass

            ms.next(on_started=_on_next_started)
            if eye_display is not None and eye_display.is_alive():
                try:
                    from src.display.eye_animator import EyeState as _ES
                    eye_display.set_music_data("Next Track", is_playing=True)
                    eye_display.set_state(_ES.MUSIC_MODE)
                except Exception:
                    pass
            return

        # 2. Volume Intents
        elif result.intent in (
            IntentType.SET_VOLUME, IntentType.INCREASE_VOLUME,
            IntentType.DECREASE_VOLUME, IntentType.MUTE_VOLUME, IntentType.UNMUTE_VOLUME
        ):
            from src.intent.volume_plugin import VolumeService
            vs = VolumeService.get_instance()
            action = result.parameters.get("action", "set")
            if result.intent == IntentType.SET_VOLUME:
                level = result.parameters.get("level", 70)
                new_vol = vs.set_volume(level)
                response = vs.format_speech_response("set", level=new_vol, is_hindi=is_hindi)
            elif result.intent == IntentType.INCREASE_VOLUME:
                new_vol = vs.increase_volume(result.parameters.get("step", 15))
                response = vs.format_speech_response("increase", level=new_vol, is_hindi=is_hindi)
            elif result.intent == IntentType.DECREASE_VOLUME:
                new_vol = vs.decrease_volume(result.parameters.get("step", 15))
                response = vs.format_speech_response("decrease", level=new_vol, is_hindi=is_hindi)
            elif result.intent == IntentType.MUTE_VOLUME:
                vs.mute()
                response = vs.format_speech_response("mute", is_hindi=is_hindi)
            else:
                new_vol = vs.get_volume()
                vs.unmute()
                response = vs.format_speech_response("unmute", level=new_vol, is_hindi=is_hindi)

            print(f"[Morphix] 🔊 Volume: {vs.get_volume()}% ({action})")
            print(f"[Morphix] 🤖 Response: {response}")
            try:
                tts.speak(response)
            except Exception as ex:
                logger.warning("TTS speaking failed for volume: %s", str(ex))
            return

        # 3. System Telemetry Intent
        elif result.intent == IntentType.GET_SYSTEM_HEALTH:
            from src.intent.system_health_plugin import SystemHealthService
            shs = SystemHealthService.get_instance()
            t_data = shs.get_telemetry()
            response = shs.format_speech_response(t_data, is_hindi=is_hindi)
            print(f"[Morphix] 🌡️ Telemetry: CPU {t_data['cpu']}% | RAM {t_data['ram']}% | Temp {t_data['temperature']}°C | Bat {t_data['battery']}%")
            print(f"[Morphix] 🤖 Response: {response}")
            if eye_display is not None and eye_display.is_alive():
                try:
                    from src.display.eye_animator import EyeState as _ES
                    eye_display.set_telemetry_data(
                        cpu=t_data["cpu"],
                        ram=t_data["ram"],
                        temp=t_data["temperature"],
                        battery=t_data["battery"],
                        is_charging=t_data["is_charging"],
                        is_alert=t_data["is_thermal_alert"]
                    )
                    if t_data.get("is_thermal_alert", False):
                        eye_display.set_state(_ES.SICK_COLD)
                    else:
                        eye_display.set_state(_ES.TELEMETRY_HUD)
                except Exception:
                    pass
            try:
                tts.speak(response)
            except Exception as ex:
                logger.warning("TTS speaking failed for telemetry: %s", str(ex))
            if eye_display is not None and eye_display.is_alive():
                try:
                    from src.display.eye_animator import EyeState as _ES
                    eye_display.set_state(_ES.LISTENING)
                except Exception:
                    pass
            return

        # 4. Weather Intent
        elif result.intent == IntentType.GET_WEATHER:
            from src.intent.weather_plugin import WeatherService
            ws = WeatherService.get_instance()
            target_city = result.parameters.get("city") if result.parameters else None
            w_data = ws.get_weather(target_city=target_city)
            response = ws.format_speech_response(w_data, is_hindi=is_hindi)
            print(f"[Morphix] 🌦️ Weather Info: {w_data['city']} | {w_data['temperature']}°C | {w_data['description']}")
            print(f"[Morphix] 🤖 Response: {response}")
            if eye_display is not None and eye_display.is_alive():
                try:
                    from src.display.eye_animator import EyeState as _ES
                    eye_display.set_weather_data(
                        city=w_data['city'],
                        temp=w_data['temperature'],
                        condition=w_data['condition'],
                        desc=w_data['description']
                    )
                    eye_display.set_state(_ES.WEATHER_HUD)
                except Exception:
                    pass
            try:
                tts.speak(response)
            except Exception as ex:
                logger.warning("TTS speaking failed for weather: %s", str(ex))
            if eye_display is not None and eye_display.is_alive():
                try:
                    from src.display.eye_animator import EyeState as _ES
                    eye_display.set_state(_ES.LISTENING)
                except Exception:
                    pass
            return

        is_todo_intent = result.intent in [
            IntentType.ADD_TODO, IntentType.SHOW_TODO, IntentType.COMPLETE_TODO,
            IntentType.DELETE_TODO, IntentType.CLEAR_TODO
        ]
        
        if is_todo_intent:
            parsed_todo = {
                "intent": result.intent.value,
                "item": result.parameters.get("task", ""),
                "old_item": result.parameters.get("old_item", ""),
                "new_item": result.parameters.get("new_item", "")
            }
            process_todo_command(parsed_todo, todo_mgr, sound_player, wake_sound, tts)
            return

        # ── Single-pass inline handler for remaining intents ──────────────────
        # IMPORTANT: Uses already-classified `result` — NO second classify() call.
        # This is the authoritative handler for Reminder, Festival, Profile,
        # Surveillance intents. `process_voice_command()` is no longer called
        # from here to prevent double-TTS and double-classification.
        response = ""
        _is_hindi = is_hindi

        if result.intent == IntentType.ADD_REMINDER:
            rem_text = result.parameters.get("text", "")
            time_expr = result.parameters.get("time_expr", "")
            if rem_text:
                reminder_mgr.add_reminder(rem_text, time_expr)
                response = "Reminder saved." if not _is_hindi else "Yaad dilana save ho gaya."
            else:
                response = "I couldn't catch the reminder details." if not _is_hindi else "Reminder ki details samajh nahi aayi."

        elif result.intent == IntentType.SHOW_REMINDERS:
            pending_rems = reminder_mgr.get_pending_reminders()
            if eye_display is not None and eye_display.is_alive():
                try:
                    rem_dicts = [{"text": r.get("message") or r.get("text", "Reminder"), "due": r.get("trigger_time", "")} for r in pending_rems]
                    if hasattr(eye_display, "set_overlay"):
                        eye_display.set_overlay("REMINDER_VIEW", data=rem_dicts, timeout=10.0)
                    elif hasattr(eye_display, "show_reminders_overlay"):
                        eye_display.show_reminders_overlay(rem_dicts, timeout=10.0)
                except Exception as ex:
                    logger.warning("Failed to render reminders overlay: %s", ex)
            if not pending_rems:
                response = "Reminder list is empty." if not _is_hindi else "Koi reminder nahi hai."
            else:
                rem_list_str = ", ".join(f"{r.get('text', '')}" for r in pending_rems)
                count = len(pending_rems)
                response = f"You have {count} pending reminder{'s' if count != 1 else ''}: {rem_list_str}."

        elif result.intent == IntentType.DELETE_REMINDER:
            target_str = result.parameters.get("target", "")
            deleted_rem = reminder_mgr.delete_reminder(target_str)
            if deleted_rem:
                response = "Reminder deleted." if not _is_hindi else "Reminder delete ho gaya."
            else:
                response = "Reminder not found." if not _is_hindi else "Reminder nahi mila."

        elif result.intent == IntentType.GET_FESTIVAL:
            from src.intent.festival_plugin import FestivalIntentPlugin
            plugin = FestivalIntentPlugin()
            response = plugin.handle(result, display=eye_display)

        elif result.intent == IntentType.SET_USER_NAME:
            from src.intent.profile_plugin import ProfileService
            ps = ProfileService.get_instance()
            target_name = result.parameters.get("name", "User")
            response = ps.set_user_name(target_name, is_hindi=_is_hindi)

        elif result.intent == IntentType.GET_USER_NAME:
            from src.intent.profile_plugin import ProfileService
            ps = ProfileService.get_instance()
            response = ps.get_user_name(is_hindi=_is_hindi)

        elif result.intent == IntentType.ENROLL_FACE:
            from src.intent.profile_plugin import ProfileService
            ps = ProfileService.get_instance()
            prompt_msg = (
                "Kripya camera ki taraf dekhein. Aapka face capture kiya jaa raha hai."
                if _is_hindi
                else "Please look directly into the camera. Capturing your face now."
            )
            print(f"[Morphix] 🤖 Response: {prompt_msg}")
            if eye_display is not None and eye_display.is_alive():
                try:
                    from src.display.eye_animator import EyeState as _ES
                    eye_display.set_state(_ES.THINKING)
                except Exception:
                    pass
            try:
                tts.speak(prompt_msg)
            except Exception:
                pass
            response = ps.enroll_user_face(is_hindi=_is_hindi)
            if eye_display is not None and eye_display.is_alive():
                try:
                    from src.display.eye_animator import EyeState as _ES
                    eye_display.set_state(_ES.HAPPY)
                except Exception:
                    pass

        elif result.intent == IntentType.START_SURVEILLANCE:
            from src.intent.surveillance_plugin import SurveillanceService
            ss = SurveillanceService.get_instance()
            if eye_display is not None and eye_display.is_alive():
                try:
                    from src.display.eye_animator import EyeState as _ES
                    eye_display.set_state(_ES.SENTRY_SCAN)
                except Exception:
                    pass
            response = ss.start_surveillance(is_hindi=_is_hindi)

        elif result.intent == IntentType.STOP_SURVEILLANCE:
            from src.intent.surveillance_plugin import SurveillanceService
            ss = SurveillanceService.get_instance()
            response = ss.stop_surveillance(is_hindi=_is_hindi)
            if eye_display is not None and eye_display.is_alive():
                try:
                    from src.display.eye_animator import EyeState as _ES
                    eye_display.set_state(_ES.IDLE_NEUTRAL)
                except Exception:
                    pass

        elif result.intent == IntentType.GET_SURVEILLANCE_STATUS:
            from src.intent.surveillance_plugin import SurveillanceService
            ss = SurveillanceService.get_instance()
            response = ss.get_status(is_hindi=_is_hindi)

        elif result.intent == IntentType.DELETE_SURVEILLANCE_CAPTURES:
            from src.intent.surveillance_plugin import SurveillanceService
            ss = SurveillanceService.get_instance()
            response = ss.delete_surveillance_captures(is_hindi=_is_hindi)

        elif result.intent == IntentType.PLAY_RPS:
            from src.intent.rps_game_plugin import RPSGamePlugin
            plugin = RPSGamePlugin()
            response = plugin.handle(result, display=eye_display, tts=tts)

        else:
            response = "I'm not sure how to do that yet." if not _is_hindi else "Mujhe abhi yeh nahi pata."

        # Single TTS call for all inline-handled intents
        if response:
            logger.info("Speaking response: '%s'", response)
            print(f"[Morphix] 🤖 Response: {response}")
            if eye_display is not None and eye_display.is_alive():
                try:
                    from src.display.eye_animator import EyeState as _ES
                    eye_display.set_state(_ES.SPEAKING)
                except Exception:
                    pass
            try:
                tts.speak(response)
            except Exception as ex:
                logger.warning("TTS speaking failed: %s", str(ex))
            if eye_display is not None and eye_display.is_alive():
                try:
                    from src.display.eye_animator import EyeState as _ES
                    eye_display.set_state(_ES.LISTENING)
                except Exception:
                    pass
        return


    # 3. Step 2: If NO local plugin matches (Unclassified / General Chat) -> Send to Groq Chat
    print("[Morphix] 🌐 Mode: GROQ AI")
    try:
        from src.ai.groq_chat import GroqChatManager
        if 'groq_chat_mgr' not in globals() or groq_chat_mgr is None:
            groq_chat_mgr = GroqChatManager()

        # Use emotion-tagged response when eye display is running
        if eye_display is not None and eye_display.is_alive():
            response, emotion_tag = groq_chat_mgr.chat_with_emotion(cleaned_command)
            logger.info("Groq emotion tag: %s", emotion_tag)

            # Smart Intent Rejector (Background / ambient chatter / third-person speech)
            if emotion_tag == "IGNORE" or response.strip() in ("[IGNORE]", ""):
                logger.info("[Morphix] 🔇 Ambient chatter / ignore tag detected from Groq. Skipping speech.")
                try:
                    from src.display.eye_animator import EyeState
                    eye_display.set_state(EyeState.CONFUSED)
                except Exception:
                    pass
                return

            # Apply emotion expression during speaking
            try:
                from src.display.eye_animator import EyeState
                if emotion_tag in EyeState.__members__:
                    eye_display.set_state(EyeState[emotion_tag])
                else:
                    eye_display.set_state(EyeState.SPEAKING)
            except Exception as ex:
                logger.debug("Could not map emotion tag '%s': %s", emotion_tag, ex)
                try:
                    from src.display.eye_animator import EyeState
                    eye_display.set_state(EyeState.SPEAKING)
                except Exception:
                    pass
        else:
            response = groq_chat_mgr.chat(cleaned_command)
            if response.strip() == "[IGNORE]" or "[IGNORE]" in response:
                logger.info("[Morphix] 🔇 Ambient chatter / ignore tag detected from Groq. Skipping speech.")
                return
    except Exception as ex:
        logger.error("Failed to run GroqChatManager: %s", str(ex))
        response = "Sorry, I had a glitch processing that!"

    print(f"[Morphix] 🤖 Response: {response}")
    try:
        tts.speak(response)
    except Exception as ex:
        logger.warning("TTS speaking failed for Groq: %s", str(ex))
    if eye_display is not None and eye_display.is_alive():
        try:
            from src.display.eye_animator import EyeState as _ES
            eye_display.set_state(_ES.LISTENING)
        except Exception:
            pass


def process_todo_command(
    parsed_todo: dict,
    todo_mgr: TodoManager,
    sound_player: SoundPlayer,
    wake_sound: str,
    tts: TextToSpeech
) -> bool:
    """
    Executes the Todo operation, plays a bark sound, and speaks the confirmation via Edge TTS (Aarti).
    Returns True if processed successfully, False if intent was UNKNOWN.
    """
    global eye_display
    intent = parsed_todo.get("intent", "UNKNOWN")
    if intent == "UNKNOWN":
        return False

    # 1. Play Bark sound
    try:
        sound_player.play_sound(wake_sound)
    except Exception as ex:
        logger.warning("Sound playback skipped or failed for todo: %s", str(ex))

    # 2. Execute operation
    item = parsed_todo.get("item", "")
    old_item = parsed_todo.get("old_item", "")
    new_item = parsed_todo.get("new_item", "")

    response = ""
    if intent == "ADD_TODO":
        try:
            response = todo_mgr.add_todo(item)
        except ValueError:
            response = "Task already exists."
    elif intent == "DELETE_TODO":
        deleted = todo_mgr.delete_todo(item)
        if deleted:
            response = f"Removed {item} from your todo list."
        else:
            response = "Todo not found."
    elif intent == "COMPLETE_TODO":
        completed = todo_mgr.complete_todo(item)
        if completed:
            response = "Todo completed."
        else:
            response = "Todo not found."
    elif intent == "EDIT_TODO":
        updated = todo_mgr.edit_todo(old_item, new_item)
        if updated:
            response = "Todo updated."
        else:
            response = "Todo not found."
    elif intent == "SHOW_TODO":
        pending = todo_mgr.get_pending_todos()
        if eye_display is not None and eye_display.is_alive():
            try:
                if hasattr(eye_display, "set_overlay"):
                    eye_display.set_overlay("TODO_VIEW", tasks=pending, timeout=10.0)
                elif hasattr(eye_display, "show_todo_overlay"):
                    eye_display.show_todo_overlay(pending, timeout=10.0)
            except Exception as ex:
                logger.warning("Failed to render todo overlay: %s", ex)
        if not pending:
            response = "Your todo list is empty!"
        else:
            task_list_str = ", ".join(f"{i+1}. {t}" for i, t in enumerate(pending))
            if len(pending) == 1:
                response = f"You have 1 pending task: {task_list_str}."
            else:
                response = f"You have {len(pending)} pending tasks: {task_list_str}."
    elif intent == "CLEAR_TODO":
        todo_mgr.clear_all_todos()
        if eye_display is not None and eye_display.is_alive():
            try:
                if hasattr(eye_display, "set_overlay"):
                    eye_display.set_overlay("TODO_VIEW", tasks=[], timeout=4.0)
                elif hasattr(eye_display, "show_todo_overlay"):
                    eye_display.show_todo_overlay([], timeout=4.0)
            except Exception as ex:
                logger.warning("Failed to render clear-todo overlay: %s", ex)
        response = "All tasks have been cleared from your todo list."

    # 3. Speak via Edge TTS (Aarti)
    print(f"[Morphix] 🤖 Response: {response}")
    if eye_display is not None and eye_display.is_alive():
        try:
            from src.display.eye_animator import EyeState as _ES
            eye_display.set_state(_ES.SPEAKING)
        except Exception:
            pass
    try:
        tts.speak(response)
    except Exception as ex:
        logger.warning("TTS speaking failed for todo: %s", str(ex))
    if eye_display is not None and eye_display.is_alive():
        try:
            from src.display.eye_animator import EyeState as _ES
            eye_display.set_state(_ES.LISTENING)
        except Exception:
            pass

    return True


def process_voice_command(
    command_text: str,
    intent_classifier: IntentClassifier,
    todo_mgr: TodoManager,
    reminder_mgr: ReminderManager,
    tts: TextToSpeech
) -> str:
    """
    Auxiliary utility that classifies and executes a voice command.

    NOTE: This function is retained for standalone utility use (e.g., tests, direct calls).
    It is NO LONGER called from the main handle_incoming_command() routing path to
    prevent double-classification and double-TTS (see inline handler block in that function).

    TodoManager API: Exclusively uses add_task/get_pending_tasks/complete_task/
    delete_task/clear_all_tasks throughout to prevent dual-API divergence (#3).
    """
    global eye_display, IntentType
    if IntentType is None:
        from src.intent.intent_classifier import IntentType

    if not command_text:
        response = "I'm not sure how to do that yet."
        if eye_display is not None and eye_display.is_alive():
            try:
                from src.display.eye_animator import EyeState as _ES
                eye_display.set_state(_ES.SPEAKING)
            except Exception:
                pass
        tts.speak(response)
        if eye_display is not None and eye_display.is_alive():
            try:
                from src.display.eye_animator import EyeState as _ES
                eye_display.set_state(_ES.LISTENING)
            except Exception:
                pass
        return response

    result = intent_classifier.classify(command_text)
    intent = result.intent
    params = result.parameters

    logger.info("Executing action for intent: %s (Params: %s)", intent.value, params)

    if intent == IntentType.ADD_TODO:
        task = params.get("task", "")
        is_hindi = params.get("is_hindi", False)
        if task:
            from src.intent.todo_plugin import TodoService
            response = TodoService.get_instance().add_task(task, is_hindi=is_hindi, display=eye_display)
        else:
            response = "I couldn't catch the task details."

    elif intent == IntentType.SHOW_TODO:
        from src.intent.todo_plugin import TodoService
        is_hindi = params.get("is_hindi", False)
        response = TodoService.get_instance().show_todo_list(is_hindi=is_hindi, display=eye_display)

    elif intent == IntentType.COMPLETE_TODO:
        task_query = params.get("task", "")
        from src.intent.todo_plugin import TodoService
        response = TodoService.get_instance().complete_task(task_query, is_hindi=params.get("is_hindi", False))

    elif intent == IntentType.DELETE_TODO:
        task_query = params.get("task", "")
        from src.intent.todo_plugin import TodoService
        response = TodoService.get_instance().delete_task(task_query, is_hindi=params.get("is_hindi", False))

    elif intent == IntentType.CLEAR_TODO:
        from src.intent.todo_plugin import TodoService
        response = TodoService.get_instance().clear_all(is_hindi=params.get("is_hindi", False), display=eye_display)

    elif intent == IntentType.ADD_REMINDER:
        rem_text = params.get("text", "")
        time_expr = params.get("time_expr", "")
        is_hindi = params.get("is_hindi", False)
        if rem_text:
            from src.intent.reminder_plugin import ReminderService
            response = ReminderService.get_instance().add_reminder(rem_text, time_expr, is_hindi=is_hindi, display=eye_display)
        else:
            response = "I couldn't catch the reminder details."

    elif intent == IntentType.SHOW_REMINDERS:
        from src.intent.reminder_plugin import ReminderService
        is_hindi = params.get("is_hindi", False)
        response = ReminderService.get_instance().show_reminders(is_hindi=is_hindi, display=eye_display)

    elif intent == IntentType.DELETE_REMINDER:
        target_str = params.get("target", "")
        from src.intent.reminder_plugin import ReminderService
        response = ReminderService.get_instance().delete_reminder(target_str, is_hindi=params.get("is_hindi", False))

    elif intent == IntentType.GET_WEATHER:
        from src.intent.weather_plugin import WeatherService
        ws = WeatherService.get_instance()
        target_city = params.get("city") if params else None
        w_data = ws.get_weather(target_city=target_city)
        is_hindi = params.get("is_hindi", False)
        response = ws.format_speech_response(w_data, is_hindi=is_hindi)
        if eye_display is not None and eye_display.is_alive():
            try:
                from src.display.eye_animator import EyeState as _ES
                eye_display.set_weather_data(
                    city=w_data['city'],
                    temp=w_data['temperature'],
                    condition=w_data['condition'],
                    desc=w_data['description']
                )
                eye_display.set_state(_ES.WEATHER_HUD)
            except Exception:
                pass

    elif intent == IntentType.PLAY_MUSIC:
        from src.intent.music_plugin import MusicService
        ms = MusicService.get_instance()
        query = params.get("query", "popular trending music")
        title = query.title()
        response = ms.format_speech_response("play", title=title, is_hindi=params.get("is_hindi", False))
        if eye_display is not None and eye_display.is_alive():
            try:
                from src.display.eye_animator import EyeState as _ES
                eye_display.set_music_data(title, is_playing=True)
                eye_display.set_state(_ES.MUSIC_MODE)
            except Exception:
                pass
        ms.play(query)

    elif intent == IntentType.NEXT_MUSIC:
        from src.intent.music_plugin import MusicService
        ms = MusicService.get_instance()
        response = ms.format_speech_response("next", is_hindi=params.get("is_hindi", False))
        if eye_display is not None and eye_display.is_alive():
            try:
                from src.display.eye_animator import EyeState as _ES
                eye_display.set_music_data("Next Track", is_playing=True)
                eye_display.set_state(_ES.MUSIC_MODE)
            except Exception:
                pass
        ms.next()

    elif intent == IntentType.PAUSE_MUSIC:
        from src.intent.music_plugin import MusicService
        ms = MusicService.get_instance()
        ms.pause()
        response = ms.format_speech_response("pause", is_hindi=params.get("is_hindi", False))

    elif intent == IntentType.RESUME_MUSIC:
        from src.intent.music_plugin import MusicService
        ms = MusicService.get_instance()
        ms.resume()
        response = ms.format_speech_response("resume", is_hindi=params.get("is_hindi", False))
        if eye_display is not None and eye_display.is_alive():
            try:
                from src.display.eye_animator import EyeState as _ES
                eye_display.set_state(_ES.MUSIC_MODE)
            except Exception:
                pass

    elif intent == IntentType.STOP_MUSIC:
        from src.intent.music_plugin import MusicService
        ms = MusicService.get_instance()
        ms.stop()
        response = ms.format_speech_response("stop", is_hindi=params.get("is_hindi", False))

    elif intent in (IntentType.SET_VOLUME, IntentType.INCREASE_VOLUME, IntentType.DECREASE_VOLUME, IntentType.MUTE_VOLUME, IntentType.UNMUTE_VOLUME):
        from src.intent.volume_plugin import VolumeService
        vs = VolumeService.get_instance()
        is_hindi = params.get("is_hindi", False)
        if intent == IntentType.SET_VOLUME:
            new_vol = vs.set_volume(params.get("level", 70))
            response = vs.format_speech_response("set", level=new_vol, is_hindi=is_hindi)
        elif intent == IntentType.INCREASE_VOLUME:
            new_vol = vs.increase_volume(params.get("step", 15))
            response = vs.format_speech_response("increase", level=new_vol, is_hindi=is_hindi)
        elif intent == IntentType.DECREASE_VOLUME:
            new_vol = vs.decrease_volume(params.get("step", 15))
            response = vs.format_speech_response("decrease", level=new_vol, is_hindi=is_hindi)
        elif intent == IntentType.MUTE_VOLUME:
            vs.mute()
            response = vs.format_speech_response("mute", is_hindi=is_hindi)
        else:
            new_vol = vs.get_volume()
            vs.unmute()
            response = vs.format_speech_response("unmute", level=new_vol, is_hindi=is_hindi)

    elif intent == IntentType.GET_SYSTEM_HEALTH:
        from src.intent.system_health_plugin import SystemHealthService
        shs = SystemHealthService.get_instance()
        t_data = shs.get_telemetry()
        response = shs.format_speech_response(t_data, is_hindi=params.get("is_hindi", False))
        if eye_display is not None and eye_display.is_alive():
            try:
                from src.display.eye_animator import EyeState as _ES
                eye_display.set_telemetry_data(
                    cpu=t_data["cpu"],
                    ram=t_data["ram"],
                    temp=t_data["temperature"],
                    battery=t_data["battery"],
                    is_charging=t_data["is_charging"],
                    is_alert=t_data["is_thermal_alert"]
                )
                if t_data.get("is_thermal_alert", False):
                    eye_display.set_state(_ES.SICK_COLD)
                else:
                    eye_display.set_state(_ES.TELEMETRY_HUD)
            except Exception:
                pass
    elif intent == IntentType.GET_FESTIVAL:
        from src.intent.festival_plugin import FestivalService
        fest_svc = FestivalService.get_instance()
        fest = fest_svc.get_todays_festival()
        is_hindi = params.get("is_hindi", False)
        if fest:
            response = fest.greeting_hi if is_hindi else fest.greeting_en
            if eye_display is not None and eye_display.is_alive():
                try:
                    eye_display.set_festival_theme(fest.theme, fest.badge_text)
                except Exception:
                    pass
        else:
            if is_hindi:
                response = "Aaj koi vishesh tyohar nahi hai, lekin har din ek naya utsav hai!"
            else:
                response = "There are no major festivals today, but every day is a celebration!"

    elif intent == IntentType.SET_USER_NAME:
        from src.intent.profile_plugin import ProfileService
        ps = ProfileService.get_instance()
        target_name = params.get("name", "User")
        is_hindi = params.get("is_hindi", False)
        response = ps.set_user_name(target_name, is_hindi=is_hindi)

    elif intent == IntentType.GET_USER_NAME:
        from src.intent.profile_plugin import ProfileService
        ps = ProfileService.get_instance()
        is_hindi = params.get("is_hindi", False)
        response = ps.get_user_name(is_hindi=is_hindi)

    elif intent == IntentType.ENROLL_FACE:
        from src.intent.profile_plugin import ProfileService
        ps = ProfileService.get_instance()
        is_hindi = params.get("is_hindi", False)

        # 1. Prompt user to look at camera
        prompt_msg = "Kripya camera ki taraf dekhein. Aapka face capture kiya jaa raha hai." if is_hindi else "Please look directly into the camera. Capturing your face now."
        print(f"[Morphix] 🤖 Response: {prompt_msg}")
        if eye_display is not None and eye_display.is_alive():
            try:
                from src.display.eye_animator import EyeState as _ES
                eye_display.set_state(_ES.THINKING)
            except Exception:
                pass
        try:
            tts.speak(prompt_msg)
        except Exception:
            pass

        # 2. Capture face frames and retrain
        response = ps.enroll_user_face(is_hindi=is_hindi)
        if eye_display is not None and eye_display.is_alive():
            try:
                from src.display.eye_animator import EyeState as _ES
                eye_display.set_state(_ES.HAPPY)
            except Exception:
                pass

    elif intent == IntentType.START_SURVEILLANCE:
        from src.intent.surveillance_plugin import SurveillanceService
        ss = SurveillanceService.get_instance()
        is_hindi = params.get("is_hindi", False)
        if eye_display is not None and eye_display.is_alive():
            try:
                from src.display.eye_animator import EyeState as _ES
                eye_display.set_state(_ES.SENTRY_SCAN)
            except Exception:
                pass
        response = ss.start_surveillance(is_hindi=is_hindi)

    elif intent == IntentType.STOP_SURVEILLANCE:
        from src.intent.surveillance_plugin import SurveillanceService
        ss = SurveillanceService.get_instance()
        is_hindi = params.get("is_hindi", False)
        response = ss.stop_surveillance(is_hindi=is_hindi)
        if eye_display is not None and eye_display.is_alive():
            try:
                from src.display.eye_animator import EyeState as _ES
                eye_display.set_state(_ES.IDLE_NEUTRAL)
            except Exception:
                pass

    elif intent == IntentType.GET_SURVEILLANCE_STATUS:
        from src.intent.surveillance_plugin import SurveillanceService
        ss = SurveillanceService.get_instance()
        is_hindi = params.get("is_hindi", False)
        response = ss.get_status(is_hindi=is_hindi)

    elif intent == IntentType.DELETE_SURVEILLANCE_CAPTURES:
        from src.intent.surveillance_plugin import SurveillanceService
        ss = SurveillanceService.get_instance()
        is_hindi = params.get("is_hindi", False)
        response = ss.delete_surveillance_captures(is_hindi=is_hindi)

    elif intent == IntentType.PLAY_RPS:
        from src.intent.rps_game_plugin import RPSService
        rps_svc = RPSService.get_instance()
        is_hindi = params.get("is_hindi", False)
        response = rps_svc.play_round(is_hindi=is_hindi, display=eye_display, tts=tts)

    else:
        response = "I'm not sure how to do that yet."

    logger.info("Speaking response: '%s'", response)
    print(f"[Morphix] 🤖 Response: {response}")
    if eye_display is not None and eye_display.is_alive():
        try:
            from src.display.eye_animator import EyeState as _ES
            eye_display.set_state(_ES.SPEAKING)
        except Exception:
            pass
    try:
        tts.speak(response)
    except Exception as ex:
        logger.warning("TTS speaking failed: %s", str(ex))
    if eye_display is not None and eye_display.is_alive():
        try:
            from src.display.eye_animator import EyeState as _ES
            eye_display.set_state(_ES.LISTENING)
        except Exception:
            pass
    return response


def main() -> None:
    """
    Main application entrypoint for Morphix Voice Assistant.
    Coordinates continuous pre-buffered microphone stream, RapidFuzz wake word matching,
    Intent classification, offline Todo & Reminder management, and Edge TTS (Aarti) voice responses.

    Supports Config.DEVELOPMENT_MODE = True to bypass wake-word requirement during dev phases.
    """
    # 1. Load environment variables strictly from .env before doing any imports or NLU initialization
    import os
    from dotenv import load_dotenv
    env_exists = os.path.exists(".env")
    if env_exists:
        # override=True ensures .env values always take precedence over OS-level env vars
        load_dotenv(dotenv_path=".env", override=True)
    else:
        print("\n[ERROR] .env file not found.")
        print("Please create a .env file before running Morphix.\n")

    # Declare globals and import modules so they receive loaded environment variables
    global Config, SoundPlayer, IntentClassifier, IntentType, ReminderChecker, ReminderManager
    global TodoManager, TranscriptCleaner, ReminderIntentParser, TodoParser, AIRegistry
    global SpeechToText, TextToSpeech, VoiceAssistant, WakeWordDetector, OpenWakeWordEngine, WakeWordManager, groq_chat_mgr, eye_display

    from config import Config
    from src.audio.sound_player import SoundPlayer
    from src.intent.intent_classifier import IntentClassifier, IntentType
    from src.reminder.reminder_checker import ReminderChecker
    from src.reminder.reminder_manager import ReminderManager
    from src.todo.todo_manager import TodoManager
    from src.nlp.transcript_cleaner import TranscriptCleaner
    from src.intent.reminder_parser import ReminderIntentParser
    from src.intent.todo_intent import TodoIntentParser as TodoParser
    from src.ai.ai_registry import AIRegistry
    from src.voice.speech_to_text import SpeechToText
    from src.voice.text_to_speech import TextToSpeech
    from src.voice.voice_assistant import VoiceAssistant
    from src.voice.wake_word import WakeWordDetector
    from src.wakeword.wakeword_engine import OpenWakeWordEngine
    from src.wakeword.wakeword_manager import WakeWordManager

    # Load validate_api_key dynamically
    from src.gemini.gemini_client import validate_api_key

    # Validate API key
    if env_exists:
        api_key = os.getenv("GEMINI_API_KEY")
        is_valid, reason = validate_api_key(api_key)
    else:
        api_key = None
        is_valid = False
        reason = ".env file not found"

    # 2. Print Startup NLU Diagnostics
    print("==================================================")
    print("        MORPHIX AI DESKTOP COMPANION ROBOT        ")
    print("           Real Offline Voice Assistant           ")
    print("==================================================")
    
    if env_exists:
        print(f"Environment file loaded: {os.path.abspath('.env')}")
    else:
        print("Environment file loaded: None")

    if api_key:
        masked_key = api_key[:6] + "..." if len(api_key) > 6 else "..."
        print(f"API key detected: Yes ({masked_key})")
    else:
        print("API key detected: No")

    if is_valid:
        print("Gemini: Enabled (will validate connection on first request)")
    else:
        print(f"Gemini: Disabled — {reason}")
        if api_key and not is_valid:
            print("  → Get a valid API key at: https://aistudio.google.com/app/apikey")
    print("==================================================")

    transition_state(AssistantState.INITIALIZING)
    print("Initializing components...")

    # -----------------------------------------------------------------------
    # Start OLED Eye Display (daemon thread — starts immediately so the user
    # sees the eyes as Morphix boots up).
    # -----------------------------------------------------------------------
    try:
        from src.display.eye_animator import EyeAnimatorThread, EyeState
        eye_display = EyeAnimatorThread(title="Morphix — OLED Eyes")
        ok = eye_display.start(wait_ready=True, timeout=5.0)
        if ok:
            logger.info("Eye display started successfully.")
            eye_display.set_state(EyeState.IDLE_NEUTRAL)

            # Check active Indian Festival / seasonal celebration for today
            try:
                from src.intent.festival_plugin import FestivalService
                fest = FestivalService.get_instance().get_todays_festival()
                if fest:
                    logger.info("🎉 Today is %s! Applying OLED theme: %s", fest.name, fest.theme)
                    eye_display.set_festival_theme(fest.theme, fest.badge_text)
            except Exception as _fest_exc:
                logger.debug("Festival startup check exception: %s", _fest_exc)

            # Register CameraManager display state callback for SENTRY_SCAN
            try:
                from src.vision.camera_manager import CameraManager
                from src.display.eye_animator import EyeState as _ES

                def _on_camera_eye_state(state_name: str):
                    if eye_display and eye_display.is_alive():
                        if state_name == "SENTRY_SCAN":
                            eye_display.set_state(_ES.SENTRY_SCAN)
                        elif state_name == "IDLE_NEUTRAL":
                            eye_display.set_state(_ES.IDLE_NEUTRAL)

                CameraManager.get_instance().set_display_callback(_on_camera_eye_state)
            except Exception as _cam_exc:
                logger.debug("Camera callback registration exception: %s", _cam_exc)
        else:
            logger.warning("Eye display did not start cleanly — continuing without it.")
            eye_display = None
    except Exception as _eye_exc:
        logger.warning("Could not start eye display: %s", str(_eye_exc))
        eye_display = None

    dev_mode = getattr(Config, "DEVELOPMENT_MODE", False)
    debug_pipeline = getattr(Config, "DEBUG_AUDIO_PIPELINE", False)

    sound_player = SoundPlayer()
    tts = TextToSpeech()
    stt = SpeechToText()

    # Initialize Phase 1 Todo, Reminder, and Intent Classifier modules
    todo_mgr = TodoManager()
    reminder_mgr = ReminderManager()
    intent_classifier = IntentClassifier()

    # Initialize modern AI capability registry with lazy initialization
    ai_registry = AIRegistry()

    # Register factory for Gemini (lazy loaded)
    def make_gemini_engine():
        from src.gemini.gemini_client import GeminiClient
        from src.gemini.chat_memory import ChatMemory
        from src.gemini.gemini_intent_engine import GeminiIntentEngine
        
        g_client = GeminiClient()
        memory = ChatMemory()
        return GeminiIntentEngine(client=g_client, memory=memory)

    ai_registry.register_provider("Gemini", make_gemini_engine)

    # Construct the Gemini Router only if .env exists and NLU is valid/enabled
    router = None
    if is_valid:
        try:
            gemini_engine = ai_registry.get_provider("Gemini")
            from src.gemini.intent_router import IntentRouter
            router = IntentRouter(todo_mgr=todo_mgr, reminder_mgr=reminder_mgr, gemini_engine=gemini_engine)
            logger.info("Gemini Router initialized successfully.")
        except Exception as ex:
            logger.warning("Could not initialize Gemini router: %s", str(ex))

    # Start background ReminderChecker service (polls every 10 seconds)
    reminder_checker = ReminderChecker(
        reminder_manager=reminder_mgr,
        sound_player=sound_player,
        tts=tts,
        check_interval=10.0,
        wake_sound_path=getattr(Config, "WAKE_SOUND", "assets/sounds/morphix_voice.wav")
    )
    reminder_checker.start()

    # Initialize continuous microphone stream with 500ms pre-roll (800 samples = 50ms chunk for instant responsiveness)
    mic_stream = ContinuousMicrophoneStream(sample_rate=16000, blocksize=800, preroll_ms=500, sound_player=sound_player)
    if not mic_stream.start():
        logger.error("Unable to initialize continuous microphone stream. Exiting.")
        sys.exit(1)

    wake_sound = getattr(Config, "WAKE_SOUND", "assets/sounds/morphix_voice.wav")
    # Dynamic greeting — reads current primary user name from profile config
    try:
        from src.config.profile_manager import get_primary_user as _get_user
        _user_name = _get_user()
    except Exception:
        _user_name = "there"
    greeting_msg = getattr(Config, "WAKE_RESPONSE", f"Hello {_user_name}! I'm listening.")

    # WakeWordDetector using WAKE_PHRASES configuration
    wake_phrases = getattr(Config, "WAKE_PHRASES", ["hey morphix"])
    detector = WakeWordDetector(
        wake_phrases=wake_phrases,
        threshold=getattr(Config, "SIMILARITY_THRESHOLD", 0.85)
    )

    if dev_mode:
        print("\n--------------------------------------------------")
        print("  NOTICE: DEVELOPMENT MODE ACTIVE (Wake-Word Bypassed)")
        print("  Workflow: Bark Sound -> Edge TTS Greeting -> Continuous Command Loop")
        print("--------------------------------------------------\n")

        try:
            # Simulate transition from WAIT_WAKE to ACTIVE_MODE in dev mode
            transition_state(AssistantState.ACTIVE_MODE)
            logger.info("Active Mode Started")

            # Show LISTENING eye immediately in dev mode
            if eye_display:
                try:
                    eye_display.set_state(EyeState.LISTENING)
                except Exception:
                    pass

            try:
                sound_player.play_sound(wake_sound, blocking=False)
            except Exception as ex:
                logger.warning("Sound playback skipped or unavailable: %s", str(ex))

            print("Voice Assistant Ready (Development Mode).")
            print("Press Ctrl+C at any time to stop.\n")

            active_timeout = getattr(Config, "ACTIVE_MODE_TIMEOUT", 15.0)
            last_speech_time = time.time()

            # For dev mode, simulate bark play and listen status print
            print("[Morphix] 🐶 Bark! Active listening (15s window)...")

            while True:
                if debug_pipeline:
                    print("\nNow listening for command...")
                command_wav, recognized_command, timeout_reached, updated_last_speech_time = record_command_with_vad(
                    mic_stream=mic_stream,
                    stt=stt,
                    max_duration=10.0,
                    silence_timeout=1.5,
                    post_tts_pause_ms=50,
                    active_timeout=active_timeout,
                    last_speech_time_ref=last_speech_time
                )
                # Advance inactivity reference: fixes pass-by-value timer bug
                last_speech_time = updated_last_speech_time

                if timeout_reached:
                    transition_state(AssistantState.WAIT_WAKE)
                    print("[Morphix] 😴 Waiting for wake word...")
                    # Transition to DEEP_SLEEP with floating Zzz particles
                    if eye_display:
                        try:
                            eye_display.set_state(EyeState.DEEP_SLEEP)
                        except Exception:
                            pass
                    break

                cleaned_command = TranscriptCleaner.clean(recognized_command)
                if is_meaningful_speech(cleaned_command):
                    transition_state(AssistantState.COMMAND_RECEIVED)
                    print(f"[Morphix] 👂 Heard: {cleaned_command}")

                    # THINKING eye while processing command
                    if eye_display:
                        try:
                            eye_display.set_state(EyeState.THINKING)
                        except Exception:
                            pass

                    handle_incoming_command(
                        cleaned_command=cleaned_command,
                        raw_transcript=recognized_command,
                        reminder_mgr=reminder_mgr,
                        todo_mgr=todo_mgr,
                        sound_player=sound_player,
                        wake_sound=wake_sound,
                        tts=tts,
                        router=router,
                        ai_registry=ai_registry,
                        intent_classifier=intent_classifier
                    )

                    last_speech_time = time.time()  # Reset after command processed
                    transition_state(AssistantState.ACTIVE_MODE)
                    # Return to MUSIC_MODE if music is playing, otherwise LISTENING
                    if eye_display:
                        try:
                            from src.intent.music_plugin import MusicService
                            if MusicService.get_instance().is_playing():
                                track = MusicService.get_instance().get_now_playing()
                                title = track.title if track else "Music"
                                eye_display.set_music_data(title, is_playing=True)
                                eye_display.set_state(EyeState.MUSIC_MODE)
                            else:
                                eye_display.set_state(EyeState.LISTENING)
                        except Exception:
                            pass
                else:
                    logger.info("No speech command detected.")

                time.sleep(0.1)

        except KeyboardInterrupt:
            reminder_checker.stop()
            mic_stream.stop()
            if eye_display:
                try:
                    eye_display.stop()
                except Exception:
                    pass
            print("\n\nExiting Morphix Voice Assistant (Development Mode). Goodbye!")
            sys.exit(0)

    else:
        # Standard Wake-Word Workflow
        openwakeword_engine = OpenWakeWordEngine()
        wake_word_manager = WakeWordManager(engine=openwakeword_engine)
        is_model_loaded = wake_word_manager.initialize()

        if not is_model_loaded:
            logger.warning("Custom Morphix wake-word model not found.")
            print("\n--------------------------------------------------")
            print("  NOTICE: Custom Morphix wake-word model not found.")
            print("  Using Fallback Engine with 500ms pre-speech buffer & RapidFuzz matching.")
            print("--------------------------------------------------\n")

        print("Voice Assistant Ready (Dedicated Wake-Word Mode).")
        print(f"Target Wake Phrases: {wake_phrases} (Similarity Threshold: {detector._threshold / 100.0:.2f})")
        print("Press Ctrl+C at any time to stop.\n")

        wake_word_manager.start()

        # Clear ring buffer to start fresh
        mic_stream.clear()

        last_check_time = time.time()
        transition_state(AssistantState.WAIT_WAKE)
        print("[Morphix] 😴 Waiting for wake word...")

        # Show IDLE_NEUTRAL while waiting for wake word
        if eye_display:
            try:
                eye_display.set_state(EyeState.IDLE_NEUTRAL)
            except Exception:
                pass

        try:
            while True:
                current_time = time.time()
                # Run sliding window wake word checking every 0.8 seconds
                if current_time - last_check_time >= 0.8:
                    last_check_time = current_time

                    # Extract the latest 3.0 seconds from the ring buffer
                    wav_bytes = mic_stream.get_last_n_seconds(seconds=3.0)

                    # Only run detection if we have acquired enough audio
                    if len(wav_bytes) > 1000:
                        is_wake = wake_word_manager.detect(wav_bytes)
                        if is_wake:
                            logger.info("Wake phrase detected!")

                            # Flush the buffer to prevent immediate duplicate triggers
                            mic_stream.clear()

                            # LISTENING eye on wake word
                            if eye_display:
                                try:
                                    eye_display.set_state(EyeState.LISTENING)
                                except Exception:
                                    pass

                            # Play the dog bark sound (.wav) immediately via SoundPlayer asynchronously
                            try:
                                sound_player.play_sound(wake_sound, blocking=False)
                            except Exception as ex:
                                logger.warning("Sound playback skipped or unavailable: %s", str(ex))

                            # Check if today is an Indian Festival and first wake interaction of the day
                            try:
                                from src.intent.festival_plugin import FestivalService
                                fest_svc = FestivalService.get_instance()
                                fest_today = fest_svc.get_todays_festival()
                                if fest_today and fest_svc.should_play_first_wake_greeting(fest_today):
                                    logger.info("First wake of festival day: '%s'! Speaking festive greeting.", fest_today.name)
                                    if eye_display is not None and eye_display.is_alive():
                                        try:
                                            from src.display.eye_animator import EyeState as _ES
                                            eye_display.set_state(_ES.SPEAKING)
                                        except Exception:
                                            pass
                                    tts.speak(fest_today.greeting_hi)
                                    if eye_display is not None and eye_display.is_alive():
                                        try:
                                            from src.display.eye_animator import EyeState as _ES
                                            eye_display.set_state(_ES.LISTENING)
                                        except Exception:
                                            pass
                            except Exception as _fest_err:
                                logger.debug("Festival first wake greeting exception: %s", _fest_err)

                            print("[Morphix] 🐶 Bark! Active listening (15s window)...")

                            # Standard Active mode (listens for 15 seconds for a command)
                            transition_state(AssistantState.ACTIVE_MODE)
                            logger.info("Active Mode Started")

                            active_timeout = getattr(Config, "ACTIVE_MODE_TIMEOUT", 15.0)
                            last_speech_time = time.time()

                            while True:
                                if debug_pipeline:
                                    print("\nNow listening for command...")
                                command_wav, recognized_command, timeout_reached, updated_last_speech_time = record_command_with_vad(
                                    mic_stream=mic_stream,
                                    stt=stt,
                                    max_duration=10.0,
                                    silence_timeout=1.5,
                                    post_tts_pause_ms=50,
                                    active_timeout=active_timeout,
                                    last_speech_time_ref=last_speech_time
                                )
                                # Advance inactivity reference: fixes pass-by-value timer bug
                                last_speech_time = updated_last_speech_time

                                if timeout_reached:
                                    logger.info("Active mode timeout reached. Transitioning to Sleep/Idle.")
                                    # Transition to DEEP_SLEEP with floating Zzz particles
                                    if eye_display:
                                        try:
                                            eye_display.set_state(EyeState.DEEP_SLEEP)
                                        except Exception:
                                            pass
                                    break

                                cleaned_command = TranscriptCleaner.clean(recognized_command)
                                if is_meaningful_speech(cleaned_command):
                                    transition_state(AssistantState.COMMAND_RECEIVED)
                                    print(f"[Morphix] 👂 Heard: {cleaned_command}")

                                    # THINKING eye while processing
                                    if eye_display:
                                        try:
                                            eye_display.set_state(EyeState.THINKING)
                                        except Exception:
                                            pass

                                    handle_incoming_command(
                                        cleaned_command=cleaned_command,
                                        raw_transcript=recognized_command,
                                        reminder_mgr=reminder_mgr,
                                        todo_mgr=todo_mgr,
                                        sound_player=sound_player,
                                        wake_sound=wake_sound,
                                        tts=tts,
                                        router=router,
                                        ai_registry=ai_registry,
                                        intent_classifier=intent_classifier
                                    )

                                    last_speech_time = time.time()  # Reset after command processed
                                    transition_state(AssistantState.ACTIVE_MODE)
                                    # Return to SENTRY_SCAN if surveillance active, MUSIC_MODE if music is playing, otherwise LISTENING
                                    if eye_display:
                                        try:
                                            from src.vision.camera_manager import CameraManager
                                            from src.intent.music_plugin import MusicService
                                            if CameraManager.get_instance().is_surveillance_active():
                                                eye_display.set_state(EyeState.SENTRY_SCAN)
                                            elif MusicService.get_instance().is_playing():
                                                track = MusicService.get_instance().get_now_playing()
                                                title = track.title if track else "Music"
                                                eye_display.set_music_data(title, is_playing=True)
                                                eye_display.set_state(EyeState.MUSIC_MODE)
                                            else:
                                                eye_display.set_state(EyeState.LISTENING)
                                        except Exception:
                                            pass
                                else:
                                    logger.info("No speech command detected.")

                                time.sleep(0.1)

                            # Seamlessly transition back to continuous WAIT_WAKE streaming
                            transition_state(AssistantState.WAIT_WAKE)
                            print("[Morphix] 😴 Waiting for wake word...")
                            mic_stream.clear()
                            last_check_time = time.time()
                            # Keep DEEP_SLEEP with floating Zzz particles while waiting for next wake word (unless surveillance is active)
                            if eye_display:
                                try:
                                    from src.vision.camera_manager import CameraManager
                                    if CameraManager.get_instance().is_surveillance_active():
                                        eye_display.set_state(EyeState.SENTRY_SCAN)
                                    else:
                                        eye_display.set_state(EyeState.DEEP_SLEEP)
                                except Exception:
                                    pass

        except KeyboardInterrupt:
            try:
                from src.vision.camera_manager import CameraManager
                if CameraManager.get_instance().is_surveillance_active():
                    CameraManager.get_instance().stop_surveillance()
            except Exception:
                pass
            reminder_checker.stop()
            mic_stream.stop()
            if eye_display:
                try:
                    eye_display.stop()
                except Exception:
                    pass
            print("\n\nExiting Morphix Voice Assistant. Goodbye!")
            sys.exit(0)


if __name__ == "__main__":
    main()


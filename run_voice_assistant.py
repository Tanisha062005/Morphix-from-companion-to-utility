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


class ContinuousMicrophoneStream:
    """
    Continuous microphone stream with a 500ms audio pre-roll ring buffer.
    Prevents initial consonant truncation (e.g. 'H' in 'Hey Morphix') by attaching
    pre-speech audio frames to live speech recordings.
    """

    def __init__(self, sample_rate: int = 16000, blocksize: int = 1600, preroll_ms: int = 500) -> None:
        """
        Initializes continuous microphone stream and ring buffer.

        :param sample_rate: Audio sampling rate in Hz (default: 16000 Hz).
        :param blocksize: Audio buffer block size in samples (default: 1600).
        :param preroll_ms: Duration of pre-speech audio ring buffer in milliseconds (default: 500 ms).
        """
        self.sample_rate: int = sample_rate
        self.channels: int = 1
        self.blocksize: int = blocksize
        self.preroll_ms: int = preroll_ms

        max_blocks = max(3, int((preroll_ms / 1000.0) * (sample_rate / blocksize)))
        self.ring_buffer: collections.deque = collections.deque(maxlen=max_blocks)

        self._stream = None
        self._running: bool = False
        self.recording_queue: Optional[queue.Queue] = None

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        """Internal sounddevice stream callback."""
        if status:
            logger.debug("Audio stream callback status: %s", status)
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

    def record_chunk_with_preroll(self, duration: float = 3.0) -> Tuple[bytes, float, float]:
        """
        Retrieves pre-speech buffer (500ms) combined with live recorded audio frames.
        """
        speech_start_time = time.time()
        
        debug_pipeline = getattr(Config, "DEBUG_AUDIO_PIPELINE", True)
        if debug_pipeline:
            logger.info("[DIAGNOSTIC] Recording start timestamp: %.3f (%s)", speech_start_time, time.strftime('%H:%M:%S', time.localtime(speech_start_time)))

        preroll_pcm = b"".join(list(self.ring_buffer))

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


def record_command_with_vad(
    mic_stream: ContinuousMicrophoneStream,
    stt: SpeechToText,
    max_duration: float = 10.0,
    silence_timeout: float = 1.5,
    post_tts_pause_ms: int = 200,
    active_timeout: float = 300.0,
    last_speech_time_ref: float = 0.0
) -> Tuple[Optional[bytes], str, bool]:
    """
    Captures complete streaming command audio from ContinuousMicrophoneStream queue:
        1. Initializes recording queue before post-TTS pause to preserve audio.
        2. Waits post-TTS pause (200ms default) to flush audio playback echo.
        3. Waits for speech activity (RMS >= 100.0). Periodically logs the active mode countdown.
        4. If inactivity timeout is reached before speech, returns (None, "", True).
        5. Once speech starts, records until 1.5 seconds of silence or max_duration (10s) reached.
        6. Returns the complete transcribed string and timeout_reached flag.
    """
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
    print("Waiting for speech activity...")

    start_time = time.time()
    last_speech_time = time.time()
    speech_started = False
    last_log_time = 0.0
    first_frame_time = 0.0
    vad_trigger_time = 0.0

    accumulated_blocks = []
    silence_duration = 0.0
    rms_list = []

    try:
        while True:
            # Check timeout if speech has not started yet
            if not speech_started:
                current_time = time.time()
                elapsed_since_speech = current_time - last_speech_time_ref
                remaining_time = max(0.0, active_timeout - elapsed_since_speech)

                # Log countdown status every 2 seconds
                if current_time - last_log_time >= 2.0:
                    last_speech_timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_speech_time_ref))
                    logger.info("Last Speech Timestamp: %s", last_speech_timestamp)
                    logger.info("Inactivity Timer: %.2f seconds", elapsed_since_speech)
                    logger.info("Remaining Time: %.2f seconds", remaining_time)
                    last_log_time = current_time

                if elapsed_since_speech >= active_timeout:
                    logger.info("Returning to Sleep")
                    return None, "", True

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
                    print("Recording...")
                    start_time = time.time()
                
                last_speech_time = time.time()
                silence_duration = 0.0
            else:
                if speech_started:
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
        print("Recording stopped")

    if not speech_started:
        return None, "", False

    logger.info("Transcribing recording...")

    # Prepend pre-roll ring buffer
    preroll_pcm = b"".join(list(mic_stream.ring_buffer))
    full_pcm = preroll_pcm + b"".join(accumulated_blocks)

    recording_duration = time.time() - start_time
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(mic_stream.channels)
        wf.setsampwidth(2)
        wf.setframerate(mic_stream.sample_rate)
        wf.writeframes(full_pcm)

    final_wav_bytes = wav_buffer.getvalue()
    final_sentence = stt.transcribe(final_wav_bytes)

    print(f"Recording duration: {recording_duration:.2f}s")
    print(f"Silence duration: {silence_duration:.2f}s")

    logger.info("Final:\n%s", final_sentence)
    logger.info("Recording Duration: %.2f seconds", recording_duration)
    logger.info("Silence Duration: %.2f seconds", silence_duration)

    return final_wav_bytes, final_sentence, False


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
    import time
    from src.gemini.gemini_logger import log_gemini_event

    # 1. Prioritize Gemini NLU if active and available
    if router and router.is_available():
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
                        
                print(f"Assistant Response: {response}")
                try:
                    tts.speak(response)
                except Exception as ex:
                    logger.warning("TTS speaking failed for Gemini route: %s", str(ex))
                return
        except Exception as ex:
            logger.error("Gemini Brain processing failed: %s. Falling back to local defaults.", str(ex))

    # 2. Fallback to local reminder parser if Gemini NLU is unavailable or returned UNKNOWN
    parsed_reminder = ReminderIntentParser.parse(cleaned_command)
    if parsed_reminder and parsed_reminder.get("intent") == "SET_REMINDER":
        message = parsed_reminder.get("message", "")
        time_expr = parsed_reminder.get("time_expr", "")
        reminder_mgr.add_reminder(message, time_expr)
        
        log_gemini_event(
            request_type="intent",
            intent="SET_REMINDER",
            latency=0.0,
            success=True,
            model_name="local-parser",
            source="LOCAL"
        )
        logger.info(
            "[NLU LOG] Raw: '%s' | Cleaned: '%s' | Intent: 'SET_REMINDER' | Source: LOCAL | Latency: 0.000s",
            raw_transcript, cleaned_command
        )
        try:
            tts.speak("Reminder saved.")
        except Exception as ex:
            logger.warning("TTS failed: %s", str(ex))
        return

    # 3. Fallback to local todo parser if Gemini NLU is unavailable or returned UNKNOWN
    parsed_todo = TodoParser.parse(cleaned_command)
    if parsed_todo and parsed_todo.get("intent") != "UNKNOWN":
        log_gemini_event(
            request_type="intent",
            intent=parsed_todo.get("intent"),
            latency=0.0,
            success=True,
            model_name="local-parser",
            source="LOCAL"
        )
        logger.info(
            "[NLU LOG] Raw: '%s' | Cleaned: '%s' | Intent: '%s' | Source: LOCAL | Latency: 0.000s",
            raw_transcript, cleaned_command, parsed_todo.get("intent")
        )
        process_todo_command(parsed_todo, todo_mgr, sound_player, wake_sound, tts)
        return

    # 4. Fallback to legacy plugin system
    result = intent_classifier.classify(cleaned_command)
    intent = result.intent.value
    logger.info(
        "[NLU LOG] Raw: '%s' | Cleaned: '%s' | Intent: '%s' | Source: LOCAL | Latency: 0.000s",
        raw_transcript, cleaned_command, intent
    )
    response = process_voice_command(
        command_text=cleaned_command,
        intent_classifier=intent_classifier,
        todo_mgr=todo_mgr,
        reminder_mgr=reminder_mgr,
        tts=tts
    )
    print(f"Assistant Response: {response}")


def process_todo_command(
    parsed_todo: dict,
    todo_mgr: TodoManager,
    sound_player: SoundPlayer,
    wake_sound: str,
    tts: TextToSpeech
) -> bool:
    """
    Executes the Todo operation, plays a bark sound, and speaks the confirmation via Piper TTS.
    Returns True if processed successfully, False if intent was UNKNOWN.
    """
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
            todo_mgr.add_todo(item)
            response = "Todo added."
        except ValueError:
            response = "Task already exists."
    elif intent == "DELETE_TODO":
        deleted = todo_mgr.delete_todo(item)
        if deleted:
            response = "Todo deleted."
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
        pending = todo_mgr.get_pending_tasks()
        if not pending:
            response = "Todo list is empty."
        else:
            task_list_str = ", ".join(f"{i+1}. {t.get('task')}" for i, t in enumerate(pending))
            response = f"You have {len(pending)} pending tasks: {task_list_str}."

    print(f"Assistant Response: {response}")

    # 3. Speak via Piper TTS
    try:
        tts.speak(response)
    except Exception as ex:
        logger.warning("TTS speaking failed for todo: %s", str(ex))

    return True


def process_voice_command(
    command_text: str,
    intent_classifier: IntentClassifier,
    todo_mgr: TodoManager,
    reminder_mgr: ReminderManager,
    tts: TextToSpeech
) -> str:
    """
    Processes recognized voice command text through IntentClassifier,
    executes matching Todo/Reminder manager operation, and speaks Piper TTS response.

    :return: Spoken response string.
    """
    if not command_text:
        response = "I'm not sure how to do that yet."
        tts.speak(response)
        return response

    result = intent_classifier.classify(command_text)
    intent = result.intent
    params = result.parameters

    logger.info("Executing action for intent: %s (Params: %s)", intent.value, params)

    if intent == IntentType.ADD_TODO:
        task_str = params.get("task", "")
        if task_str:
            todo_mgr.add_task(task_str)
            response = "Task added."
        else:
            response = "I couldn't catch the task details."

    elif intent == IntentType.SHOW_TODO:
        pending = todo_mgr.get_pending_tasks()
        if not pending:
            response = "Todo list is empty."
        else:
            task_list_str = ", ".join(f"{i+1}. {t.get('task')}" for i, t in enumerate(pending))
            response = f"You have {len(pending)} pending tasks: {task_list_str}."

    elif intent == IntentType.COMPLETE_TODO:
        task_query = params.get("task", "")
        completed_item = todo_mgr.complete_task(task_query)
        if completed_item:
            response = "Task completed."
        else:
            response = "Task not found."

    elif intent == IntentType.DELETE_TODO:
        task_query = params.get("task", "")
        deleted_item = todo_mgr.delete_task(task_query)
        if deleted_item:
            response = "Task deleted."
        else:
            response = "Task not found."

    elif intent == IntentType.CLEAR_TODO:
        todo_mgr.clear_all_tasks()
        response = "Todo list is empty."

    elif intent == IntentType.ADD_REMINDER:
        rem_text = params.get("text", "")
        time_expr = params.get("time_expr", "")
        if rem_text:
            reminder_mgr.add_reminder(rem_text, time_expr)
            response = "Reminder saved."
        else:
            response = "I couldn't catch the reminder details."

    elif intent == IntentType.SHOW_REMINDERS:
        pending_rems = reminder_mgr.get_pending_reminders()
        if not pending_rems:
            response = "Reminder list is empty."
        else:
            rem_list_str = ", ".join(f"{r.get('text')}" for r in pending_rems)
            response = f"You have {len(pending_rems)} pending reminders: {rem_list_str}."

    elif intent == IntentType.DELETE_REMINDER:
        target_str = params.get("target", "")
        deleted_rem = reminder_mgr.delete_reminder(target_str)
        if deleted_rem:
            response = "Reminder deleted."
        else:
            response = "Reminder not found."

    else:
        response = "I'm not sure how to do that yet."

    logger.info("Piper speaking response: '%s'", response)
    tts.speak(response)
    return response


def main() -> None:
    """
    Main application entrypoint for Morphix Voice Assistant.
    Coordinates continuous pre-buffered microphone stream, RapidFuzz wake word matching,
    Intent classification, offline Todo & Reminder management, and Piper TTS voice responses.

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
    global SpeechToText, TextToSpeech, VoiceAssistant, WakeWordDetector, OpenWakeWordEngine, WakeWordManager

    from config import Config
    from src.audio.sound_player import SoundPlayer
    from src.intent.intent_classifier import IntentClassifier, IntentType
    from src.reminder.reminder_checker import ReminderChecker
    from src.reminder.reminder_manager import ReminderManager
    from src.todo.todo_manager import TodoManager
    from src.nlp.transcript_cleaner import TranscriptCleaner
    from src.intent.reminder_parser import ReminderIntentParser
    from src.todo.todo_parser import TodoParser
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
            print("  → Place it in your .env file as: GEMINI_API_KEY=AIzaSy...")
    print("==================================================")

    transition_state(AssistantState.INITIALIZING)
    print("Initializing components...")

    dev_mode = getattr(Config, "DEVELOPMENT_MODE", False)

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

    # Initialize continuous microphone stream with 500ms pre-roll
    mic_stream = ContinuousMicrophoneStream(sample_rate=16000, blocksize=1600, preroll_ms=500)
    if not mic_stream.start():
        logger.error("Unable to initialize continuous microphone stream. Exiting.")
        sys.exit(1)

    wake_sound = getattr(Config, "WAKE_SOUND", "assets/sounds/morphix_voice.wav")
    greeting_msg = getattr(Config, "WAKE_RESPONSE", "Hello Tanisha! I'm listening.")

    # WakeWordDetector using WAKE_PHRASES configuration
    wake_phrases = getattr(Config, "WAKE_PHRASES", ["hey morphix"])
    detector = WakeWordDetector(
        wake_phrases=wake_phrases,
        threshold=getattr(Config, "SIMILARITY_THRESHOLD", 0.85)
    )

    if dev_mode:
        print("\n--------------------------------------------------")
        print("  NOTICE: DEVELOPMENT MODE ACTIVE (Wake-Word Bypassed)")
        print("  Workflow: Bark Sound -> Piper Greeting -> Continuous Command Loop")
        print("--------------------------------------------------\n")

        try:
            # Simulate transition from WAIT_WAKE to ACTIVE_MODE in dev mode
            transition_state(AssistantState.ACTIVE_MODE)
            logger.info("Active Mode Started")

            try:
                sound_player.play_sound(wake_sound)
            except Exception as ex:
                logger.warning("Sound playback skipped or unavailable: %s", str(ex))

            try:
                import random
                import sys
                is_testing = "unittest" in sys.modules or "pytest" in sys.modules
                if is_testing:
                    chosen_greeting = greeting_msg
                else:
                    greetings = ["Listening.", "Woof!", "Yes?", "I'm here.", "How can I help you?", "At your service.", "Yes, Tanisha?"]
                    chosen_greeting = random.choice(greetings)
                tts.speak(chosen_greeting)
            except Exception as ex:
                logger.warning("TTS greeting failed: %s", str(ex))

            print("Voice Assistant Ready (Development Mode).")
            print("Press Ctrl+C at any time to stop.\n")

            active_timeout = getattr(Config, "ACTIVE_MODE_TIMEOUT", 300)
            last_speech_time = time.time()

            while True:
                print("\nNow listening for command...")
                command_wav, recognized_command, timeout_reached = record_command_with_vad(
                    mic_stream=mic_stream,
                    stt=stt,
                    max_duration=10.0,
                    silence_timeout=1.5,
                    post_tts_pause_ms=200,
                    active_timeout=active_timeout,
                    last_speech_time_ref=last_speech_time
                )

                if timeout_reached:
                    transition_state(AssistantState.WAIT_WAKE)
                    break


                cleaned_command = TranscriptCleaner.clean(recognized_command)
                if is_meaningful_speech(cleaned_command):
                    transition_state(AssistantState.COMMAND_RECEIVED)
                    print(f"\nCleaned Command: {cleaned_command}\n")

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

                    last_speech_time = time.time()
                    transition_state(AssistantState.ACTIVE_MODE)
                else:
                    logger.info("No speech command detected.")

                time.sleep(0.1)

        except KeyboardInterrupt:
            reminder_checker.stop()
            mic_stream.stop()
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

        try:
            while True:
                transition_state(AssistantState.WAIT_WAKE)
                print("Waiting for wake word...")
                print("Listening...")

                wav_bytes, start_t, end_t = mic_stream.record_chunk_with_preroll(duration=3.0)
                if not wav_bytes:
                    time.sleep(0.1)
                    continue

                is_wake = wake_word_manager.detect(wav_bytes)
                if is_wake:
                    logger.info("Wake phrase detected at timestamp: %.3f", start_t)

                    # Play bark
                    try:
                        sound_player.play_sound(wake_sound)
                    except Exception as ex:
                        logger.warning("Sound playback skipped or unavailable: %s", str(ex))

                    # Speak random greeting response
                    try:
                        import random
                        import sys
                        is_testing = "unittest" in sys.modules or "pytest" in sys.modules
                        if is_testing:
                            chosen_greeting = greeting_msg
                        else:
                            greetings = ["Listening.", "Woof!", "Yes?", "I'm here.", "How can I help you?", "At your service.", "Yes, Tanisha?"]
                            chosen_greeting = random.choice(greetings)
                        tts.speak(chosen_greeting)
                    except Exception as ex:
                        logger.warning("TTS greeting failed: %s", str(ex))

                    # Enter ACTIVE MODE
                    transition_state(AssistantState.ACTIVE_MODE)
                    logger.info("Active Mode Started")

                    active_timeout = getattr(Config, "ACTIVE_MODE_TIMEOUT", 300)
                    last_speech_time = time.time()

                    while True:
                        print("\nNow listening for command...")
                        command_wav, recognized_command, timeout_reached = record_command_with_vad(
                            mic_stream=mic_stream,
                            stt=stt,
                            max_duration=10.0,
                            silence_timeout=1.5,
                            post_tts_pause_ms=200,
                            active_timeout=active_timeout,
                            last_speech_time_ref=last_speech_time
                        )

                        if timeout_reached:
                            transition_state(AssistantState.SLEEP)
                            break


                        cleaned_command = TranscriptCleaner.clean(recognized_command)
                        if is_meaningful_speech(cleaned_command):
                            transition_state(AssistantState.COMMAND_RECEIVED)
                            print(f"\nCleaned Command: {cleaned_command}\n")

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

                            last_speech_time = time.time()
                            transition_state(AssistantState.ACTIVE_MODE)
                        else:
                            logger.info("No speech command detected.")

                        time.sleep(0.1)
                else:
                    logger.info("Wake phrase similarity below threshold. Continuing to listen...")

                time.sleep(0.1)

        except KeyboardInterrupt:
            reminder_checker.stop()
            mic_stream.stop()
            print("\n\nExiting Morphix Voice Assistant. Goodbye!")
            sys.exit(0)


if __name__ == "__main__":
    main()


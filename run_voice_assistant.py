"""
Morphix Desktop Companion Robot - Real Voice Assistant Application
===================================================================
File: run_voice_assistant.py
Description:
    Live entrypoint for the Morphix Voice Assistant.
    Features a continuous microphone audio stream with a 500ms pre-speech ring buffer
    (pre-roll) to guarantee initial speech consonants ('H' in 'Hey') are never clipped.

    Includes multi-algorithm RapidFuzz wake word similarity matching, streaming Vosk command capture
    with partial/accumulated phrase collection, 1.5s VAD silence detection, offline rule-based Intent Classification,
    persistent local Todo and Reminder management, background ReminderChecker notification service,
    and Piper TTS voice responses.

    Supports Config.DEVELOPMENT_MODE = True to bypass wake-word requirements during feature dev:
    Immediately plays bark sound -> speaks greeting -> enters continuous command loop.

    Press Ctrl+C to terminate the application cleanly.
"""

import collections
from enum import Enum
import io
import logging
from pathlib import Path
import queue
import sys
import time
import wave
from typing import Optional, Tuple

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import Config
from src.audio.sound_player import SoundPlayer
from src.intent.intent_classifier import IntentClassifier, IntentType
from src.reminder.reminder_checker import ReminderChecker
from src.reminder.reminder_manager import ReminderManager
from src.todo.todo_manager import TodoManager
from src.voice.speech_to_text import SpeechToText
from src.voice.text_to_speech import TextToSpeech
from src.voice.voice_assistant import VoiceAssistant
from src.voice.wake_word import WakeWordDetector
from src.wakeword.wakeword_engine import OpenWakeWordEngine
from src.wakeword.wakeword_manager import WakeWordManager

# Setup application logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [VoiceAssistantApp]: %(message)s"
)
logger = logging.getLogger("VoiceAssistantApp")


class AssistantState(Enum):
    INITIALIZING = "INITIALIZING"
    WAIT_WAKE = "WAIT_WAKE"
    WAKE_DETECTED = "WAKE_DETECTED"
    BARK = "BARK"
    GREETING = "GREETING"
    COMMAND_RECORDING = "COMMAND_RECORDING"
    TRANSCRIBING = "TRANSCRIBING"


_current_state = AssistantState.INITIALIZING


def transition_state(new_state: AssistantState) -> None:
    global _current_state
    old_state = _current_state
    _current_state = new_state
    logger.info("Transitioning state: %s -> %s", old_state.name, new_state.name)
    print(f"Current application state: {new_state.name}")


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

        :param duration: Length of speech recording in seconds.
        :return: Tuple of (wav_bytes, speech_start_time, speech_end_time).
        """
        speech_start_time = time.time()
        preroll_pcm = b"".join(list(self.ring_buffer))

        self.recording_queue = queue.Queue()

        num_blocks = int((duration * self.sample_rate) / self.blocksize)
        live_blocks = []
        for _ in range(num_blocks):
            try:
                block = self.recording_queue.get(timeout=0.2)
                live_blocks.append(block)
            except queue.Empty:
                logger.warning("Audio queue underflow in record_chunk_with_preroll.")

        self.recording_queue = None
        speech_end_time = time.time()
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
    post_tts_pause_ms: int = 600
) -> Tuple[bytes, str]:
    """
    Captures complete streaming command audio from ContinuousMicrophoneStream queue:
        1. Waits 600ms post-TTS pause to flush audio playback echo.
        2. Streams PCM audio blocks into buffer.
        3. Stops recording only when 1.5 seconds of silence occurs after speech OR max_duration (10s) reached.
        4. Transcribes the full recorded WAV buffer using Whisper.cpp.
        5. Returns the complete transcribed string.

    :param mic_stream: ContinuousMicrophoneStream instance.
    :param stt: SpeechToText engine instance.
    :param max_duration: Max listening window in seconds (default: 10.0s).
    :param silence_timeout: Silence duration to end capture in seconds (default: 1.5s).
    :param post_tts_pause_ms: Delay after TTS before command capture in ms (default: 600ms).
    :return: Tuple of (final_wav_bytes, final_transcription_text).
    """
    import numpy as np

    time.sleep(post_tts_pause_ms / 1000.0)
    mic_stream.ring_buffer.clear()
    mic_stream.recording_queue = queue.Queue()

    logger.info("Command started")
    print("Recording started")
    print(f"Silence timeout: {silence_timeout}s")

    start_time = time.time()
    last_speech_time = time.time()
    speech_started = False

    accumulated_blocks = []
    silence_duration = 0.0

    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed >= max_duration:
                logger.info("Command listening window reached maximum duration (%.1fs).", max_duration)
                break

            try:
                block = mic_stream.recording_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            accumulated_blocks.append(block)

            frame = np.frombuffer(block, dtype=np.int16)
            rms = np.sqrt(np.mean(frame.astype(np.float32) ** 2)) if len(frame) > 0 else 0.0

            if rms >= 100.0:
                speech_started = True
                last_speech_time = time.time()
                silence_duration = 0.0
            else:
                if speech_started:
                    silence_duration = time.time() - last_speech_time
                    if silence_duration >= silence_timeout:
                        logger.info("Silence detected (%.1fs >= %.1fs). Ending command capture.", silence_duration, silence_timeout)
                        break
    finally:
        mic_stream.recording_queue = None
        print("Recording stopped")

    transition_state(AssistantState.TRANSCRIBING)

    recording_duration = time.time() - start_time
    full_pcm = b"".join(accumulated_blocks)
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(mic_stream.channels)
        wf.setsampwidth(2)
        wf.setframerate(mic_stream.sample_rate)
        wf.writeframes(full_pcm)

    final_wav_bytes = wav_buffer.getvalue()

    # Transcribe the whole recording using Whisper.cpp SpeechToText coordinator
    final_sentence = stt.transcribe(final_wav_bytes)

    print(f"Recording duration: {recording_duration:.2f}s")
    print(f"Silence duration: {silence_duration:.2f}s")

    logger.info("Final:\n%s", final_sentence)
    logger.info("Recording Duration: %.2f seconds", recording_duration)
    logger.info("Silence Duration: %.2f seconds", silence_duration)

    return final_wav_bytes, final_sentence


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
    print("==================================================")
    print("        MORPHIX AI DESKTOP COMPANION ROBOT        ")
    print("           Real Offline Voice Assistant           ")
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
            transition_state(AssistantState.BARK)
            try:
                sound_player.play_sound(wake_sound)
            except Exception as ex:
                logger.warning("Sound playback skipped or unavailable: %s", str(ex))

            transition_state(AssistantState.GREETING)
            try:
                tts.speak(greeting_msg)
            except Exception as ex:
                logger.warning("TTS greeting failed: %s", str(ex))

            print("Voice Assistant Ready (Development Mode).")
            print("Press Ctrl+C at any time to stop.\n")

            while True:
                transition_state(AssistantState.COMMAND_RECORDING)
                print("\nNow listening for command...")
                print("Listening...")

                command_wav, recognized_command = record_command_with_vad(
                    mic_stream=mic_stream,
                    stt=stt,
                    max_duration=10.0,
                    silence_timeout=1.5,
                    post_tts_pause_ms=600
                )

                if recognized_command:
                    print("\nRecognized Command:")
                    print(recognized_command)
                    print()

                    # (Intent Layer not executed during this phase)

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

                raw_text = stt.transcribe(wav_bytes)

                if raw_text:
                    logger.info("Speech Start Time: %.3f | Speech End Time: %.3f", start_t, end_t)

                    is_wake = detector.detect(raw_text)

                    if is_wake:
                        transition_state(AssistantState.WAKE_DETECTED)
                        logger.info("Wake phrase detected: '%s'", raw_text)

                        # transition state to BARK and play bark
                        transition_state(AssistantState.BARK)
                        sound_player.play_sound(wake_sound)

                        # transition state to GREETING and speak response
                        transition_state(AssistantState.GREETING)
                        tts.speak(greeting_msg)

                        # transition state to COMMAND_RECORDING and record command
                        transition_state(AssistantState.COMMAND_RECORDING)
                        print("\nNow listening for command...")
                        
                        command_wav, recognized_command = record_command_with_vad(
                            mic_stream=mic_stream,
                            stt=stt,
                            max_duration=10.0,
                            silence_timeout=1.5,
                            post_tts_pause_ms=600
                        )

                        if recognized_command:
                            print(f"\nRecognized Command: {recognized_command}\n")

                        # Return to wake-word mode log
                        print("Returning to wake-word mode...\n")
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


# 🤖 MORPHIX — Project Summary
> **Version:** 1.0.0 | **Language:** Python 3.10+ | **Platform:** Windows Desktop (Raspberry Pi Compatible)
> **Generated:** 2026-08-03 | **Author:** Tanisha | **For:** Technical Mentor Review

---

## 📂 1. COMPLETE FOLDER & FILE STRUCTURE

```
Morphix/
├── .env                                  # API keys and environment variables
├── .gitignore
├── README.md
├── config.py                             # Global project configuration constants
├── personality.py                        # Personality traits and mood state
├── requirements.txt                      # Python dependency declarations
├── run_voice_assistant.py                # MAIN ENTRY POINT — full pipeline orchestration
├── check_api_key.py                      # Standalone Gemini API key diagnostic tool
│
├── assets/
│   └── sounds/                           # WAV audio assets (bark, wake sound)
│
├── models/
│   ├── whisper/                          # Whisper.cpp GGUF model files (ggml-tiny.en.bin)
│   └── wakeword/                         # Custom OpenWakeWord model (morphix.tflite/.onnx)
│
├── piper/                                # Piper TTS binary and runtime DLLs
│   ├── piper.exe
│   ├── espeak-ng.dll
│   ├── onnxruntime.dll
│   ├── piper_phonemize.dll
│   └── espeak-ng-data/
│
├── storage/
│   ├── reminders.json                    # Persistent reminder storage
│   └── todo.json                         # Persistent to-do task storage
│
├── temp/
│   └── audio/                            # Transient TTS .wav output (auto-cleaned)
│
├── third_party/
│   └── whisper.cpp/                      # Whisper.cpp C++ source (reference)
│
├── tests/
│   ├── test_active_listening.py
│   ├── test_gemini_system.py
│   ├── test_intent_classifier.py
│   ├── test_personality.py
│   ├── test_reminder.py
│   ├── test_reminder_system.py
│   ├── test_todo.py
│   ├── test_todo_system.py
│   ├── test_tts.py
│   ├── test_voice_assistant.py
│   ├── test_wake_word.py
│   └── test_wakeword_engine.py
│
└── src/
    ├── ai/
    │   ├── ai_registry.py                # Singleton lazy-loading AI capability registry
    │   └── personality/
    │       └── emotion.py                # Emotion state logic (placeholder)
    │
    ├── audio/
    │   └── sound_player.py               # pygame-based WAV sound playback
    │
    ├── database/
    │   └── memory.py                     # JSON key-value persistent memory store
    │
    ├── emotions/
    │   └── expression_manager.py         # Facial expression state manager
    │
    ├── gemini/
    │   ├── chat_memory.py                # Rolling conversation history buffer
    │   ├── gemini_client.py              # Google GenAI SDK client wrapper
    │   ├── gemini_config.py              # Gemini hyperparameters and rate limit settings
    │   ├── gemini_intent_engine.py       # Core NLU engine: retry, backoff, deduplication
    │   ├── gemini_logger.py              # Gemini API call metrics logger
    │   ├── intent_router.py              # Dispatches validated Gemini intents to managers
    │   ├── json_parser.py                # Sanitizes Gemini JSON responses
    │   ├── prompt_builder.py             # Retrieves system prompts
    │   ├── prompt_versions.py            # Versioned system prompts (intent + chat)
    │   └── response_validator.py         # Strict JSON schema validator for Gemini output
    │
    ├── intent/
    │   ├── intent_classifier.py          # Plugin-based local regex intent classifier
    │   └── reminder_parser.py            # Rule-based reminder intent parser
    │
    ├── intents/
    │   └── reminder_parser.py            # (Stub - legacy placeholder)
    │
    ├── models/
    │   └── piper/                        # Piper ONNX voice model + JSON config
    │
    ├── nlp/
    │   └── transcript_cleaner.py         # STT transcript normalizer
    │
    ├── notification/
    │   └── notification_manager.py       # Pluggable notification dispatching
    │
    ├── reminder/
    │   ├── reminder_checker.py           # Background polling service for due reminders
    │   └── reminder_manager.py           # Full CRUD + timezone-aware reminder store
    │
    ├── reminders/
    │   ├── reminder.py                   # Legacy reminder logic (pre-refactor)
    │   └── todo.py                       # Legacy todo logic (pre-refactor)
    │
    ├── todo/
    │   ├── todo_manager.py               # Full CRUD to-do store with JSON persistence
    │   ├── todo_models.py                # TodoItem dataclass model
    │   └── todo_parser.py                # Rule-based to-do command parser
    │
    ├── vision/
    │   └── face_detection.py             # OpenCV-based face detection (placeholder)
    │
    ├── voice/
    │   ├── speech_engine.py              # Abstract Base Class for STT backends
    │   ├── speech_to_text.py             # STT orchestrator (delegates to WhisperEngine)
    │   ├── text_to_speech.py             # Piper TTS engine (subprocess + winsound)
    │   ├── voice_assistant.py            # High-level voice assistant state wrapper
    │   ├── wake_word.py                  # WakeWordDetector with fuzzy/RapidFuzz matching
    │   └── whisper_engine.py             # Whisper.cpp implementation via pywhispercpp
    │
    └── wakeword/
        ├── wakeword_engine.py            # Modular wake word engines (OWW + Fallback)
        └── wakeword_manager.py           # High-level manager with automatic engine fallback
```

---

## 📜 2. FILE-BY-FILE FUNCTIONALITY BREAKDOWN

### Root-Level Files

---

#### `run_voice_assistant.py` — MAIN ENTRY POINT
- **Purpose:** Full end-to-end voice assistant pipeline. Handles state machine, microphone streaming, wake word detection, VAD recording, STT, intent classification, and TTS response.
- **Key Classes / Functions:**
  - `AssistantState` (Enum) — States: INITIALIZING, WAIT_WAKE, ACTIVE_MODE, COMMAND_RECEIVED, SLEEP
  - `ContinuousMicrophoneStream` — Continuous 16kHz microphone capture with 500ms pre-roll ring buffer using sounddevice
  - `record_command_with_vad()` — RMS-based Voice Activity Detection; records until 1.5s silence or 10s max
  - `handle_incoming_command()` — 4-tier NLU fallback: Gemini → Local Reminder Parser → Local Todo Parser → Plugin Classifier
  - `process_todo_command()` — Executes todo operations and triggers TTS response
  - `process_voice_command()` — Legacy plugin-based command dispatcher
  - `main()` — Boot sequence; supports Dev Mode (no wake word) and Production Mode (wake word)
- **Dependencies:** sounddevice, numpy, collections, queue, io, wave, dotenv, all src modules
- **Status:** ✅ Working (Production-level quality)

---

#### `config.py`
- **Purpose:** Centralized global configuration for all Morphix components.
- **Key Class:** `Config` — WAKE_PHRASES, SIMILARITY_THRESHOLD, STT_MODEL, ACTIVE_MODE_TIMEOUT, DEBUG_AUDIO_PIPELINE, DEVELOPMENT_MODE
- **Status:** ✅ Working

---

#### `personality.py`
- **Purpose:** Morphix's personality identity, mood states, and dialogue message pools.
- **Key Class:** `Personality` — introduce(), greet(), goodbye(), idle_message(), error_message(), small_talk(), set_mood(), get_current_mood()
- **Status:** ✅ Working — Not yet integrated into the main pipeline

---

#### `check_api_key.py`
- **Purpose:** Standalone script to validate Gemini API key and print diagnostics.
- **Status:** ✅ Working (diagnostic tool only)

---

### src/voice/ — Voice and Audio Pipeline

---

#### `src/voice/speech_engine.py`
- **Purpose:** Abstract Base Class defining the SpeechEngine interface, decoupling STT backends from the application.
- **Key Class:** `SpeechEngine` (ABC) — abstract methods: initialize(), transcribe()
- **Status:** ✅ Working

---

#### `src/voice/whisper_engine.py`
- **Purpose:** Whisper.cpp STT engine implementation via pywhispercpp. Singleton model loading. Suppresses native C++ console noise.
- **Key Class:** `WhisperEngine(SpeechEngine)`
  - initialize() — Loads ggml-{model}.bin from models/whisper/; calls sys.exit(1) if file missing
  - transcribe(audio_source) — Accepts WAV bytes or file path; writes temp file; transcribes using n_processors=4; computes RTF
  - suppress_stdout_stderr() — Context manager that silences C++ console output
- **Dependencies:** pywhispercpp, Config, SpeechEngine
- **Status:** ✅ Working

---

#### `src/voice/speech_to_text.py`
- **Purpose:** Orchestration layer wrapping WhisperEngine. Adds listening state management.
- **Key Class:** `SpeechToText` — transcribe(), get_last_transcript(), clear_last_transcript()
- **Status:** ✅ Working

---

#### `src/voice/text_to_speech.py`
- **Purpose:** Offline TTS engine using Piper binary via subprocess. Auto-discovers piper.exe, .onnx model, and .json config. Plays audio via winsound on Windows.
- **Key Class:** `TextToSpeech`
  - speak(text) — Synthesizes speech, saves temp WAV, plays synchronously, deletes temp file
  - save_to_file(text, output_path) — Generates WAV only (no playback)
  - stop() — Purges winsound playback
  - _find_piper_exe(), _find_model_file(), _find_config_file() — Auto-discovery helpers
  - _validate_environment() — Verifies all paths before speaking
- **Dependencies:** subprocess, winsound (Windows), pathlib
- **Status:** ✅ Working

---

#### `src/voice/wake_word.py`
- **Purpose:** Fuzzy multi-algorithm wake word detector using RapidFuzz and Python difflib.
- **Key Class:** `WakeWordDetector`
  - detect(text) — Evaluates raw transcribed text and returns True/False
  - compute_similarity(text) — Returns (best_score, phrase, alias, similarity, is_detected)
  - _get_weighted_fuzzy_similarity() — Weighted token-level similarity with filler-word dampening
  - _get_non_filler_fuzzy_similarity() — Similarity ignoring filler words (hey, hi, etc.)
  - normalize_text(text) — Lowercases, strips punctuation, normalizes whitespace
- **Dependencies:** rapidfuzz, difflib, Config
- **Status:** ✅ Working

---

#### `src/voice/voice_assistant.py`
- **Purpose:** High-level voice assistant state/mode wrapper (secondary to run_voice_assistant.py).
- **Status:** ✅ Partial

---

### src/wakeword/ — Modular Wake Word Engines

---

#### `src/wakeword/wakeword_engine.py`
- **Purpose:** Defines BaseWakeWordEngine (ABC), OpenWakeWordEngine (TFLite/ONNX), and FallbackWakeWordEngine (STT + fuzzy fallback).
- **Key Classes:**
  - BaseWakeWordEngine — Abstract: initialize(), start(), stop(), is_detected(), get_model_path(), get_engine_name()
  - OpenWakeWordEngine — Loads morphix.tflite or morphix.onnx from models/wakeword/. Falls back gracefully if missing.
  - FallbackWakeWordEngine — Uses SpeechToText + WakeWordDetector for live testing when no custom model exists.
- **Dependencies:** openwakeword, numpy
- **Status:** ✅ Working (OpenWakeWord model not yet trained; auto-falls back)

---

#### `src/wakeword/wakeword_manager.py`
- **Purpose:** High-level coordinator using Dependency Injection. Auto-switches to FallbackWakeWordEngine if OpenWakeWord model is missing.
- **Key Class:** `WakeWordManager` — initialize(), detect(audio_frame)
- **Status:** ✅ Working

---

### src/gemini/ — Gemini AI Brain

---

#### `src/gemini/gemini_client.py`
- **Purpose:** Initializes the Google GenAI SDK client safely. Validates API key; performs first-time live validation on first API call.
- **Key Classes:**
  - validate_api_key(api_key) — Returns (bool, reason) without making a network call
  - GeminiClient — Wraps genai.Client() with HTTP timeout; disables itself gracefully on failure
  - is_available(), validate_key_if_first_time() — Used as guards before API calls
- **Dependencies:** google.genai, GeminiConfig
- **Status:** ✅ Working

---

#### `src/gemini/gemini_intent_engine.py`
- **Purpose:** Core NLU engine. Sends cleaned transcripts to Gemini, handles retries with exponential backoff, circuit breaker on consecutive 429s, deduplication, and confidence thresholding.
- **Key Class:** `GeminiIntentEngine`
  - process_transcript(transcript) — Extracts structured intent JSON from user speech
  - generate_chat_response(user_query) — Conversational companion reply with rolling memory
  - _is_rate_limited() — Circuit breaker check
  - _is_429_error(error), _extract_retry_after(error) — 429 detection and Retry-After parsing
- **Dependencies:** GeminiClient, ChatMemory, GeminiConfig, JsonParser, PromptBuilder, ResponseValidator
- **Status:** ✅ Working

---

#### `src/gemini/gemini_config.py`
- **Purpose:** Centralized Gemini hyperparameters.
- **Key Settings:** DEFAULT_MODEL_NAME = "gemini-flash-lite-latest", INTENT_TEMPERATURE = 0.0, CHAT_TEMPERATURE = 0.7, TIMEOUT_SECONDS = 8.0, MAX_RETRY_COUNT = 2, CONFIDENCE_THRESHOLD = 0.7, DEDUPLICATION_WINDOW_SECONDS = 3.0, RATE_LIMIT_CONSECUTIVE_THRESHOLD = 3, RATE_LIMIT_COOLDOWN_SECONDS = 60.0
- **Status:** ✅ Working

---

#### `src/gemini/intent_router.py`
- **Purpose:** Dispatches validated Gemini intent JSON dicts to TodoManager, ReminderManager, or generate_chat_response(). Returns spoken response strings for Piper TTS.
- **Key Class:** `IntentRouter` — is_available(), route(validated_intent)
- **Handles:** ADD_TODO, DELETE_TODO, EDIT_TODO, COMPLETE_TODO, SHOW_TODO, SET_REMINDER, DELETE_REMINDER, SHOW_REMINDERS, CHAT, and stubs for WEATHER, CALENDAR, VISION, MUSIC, ROBOT_CONTROL, WEB_SEARCH
- **Status:** ✅ Working

---

#### `src/gemini/chat_memory.py`
- **Purpose:** Rolling conversation context buffer. Keeps up to CHAT_HISTORY_LIMIT (10) messages for Gemini chat mode.
- **Key Class:** `ChatMemory` — add_exchange(), get_history(), clear()
- **Status:** ✅ Working

---

#### `src/gemini/prompt_builder.py`
- **Purpose:** Retrieves versioned system prompts for intent extraction and general chat.
- **Key Class:** `PromptBuilder` — get_intent_system_prompt(), get_chat_system_prompt()
- **Status:** ✅ Working

---

#### `src/gemini/prompt_versions.py`
- **Purpose:** Stores named, versioned system prompt strings.
  - PROMPT_INTENT_V1 — Instructs Gemini to return structured JSON intents; includes STT correction rules
  - PROMPT_CHAT_V1 — Sets Morphix companion personality for conversational chat (2-sentence TTS limit)
- **Status:** ✅ Working

---

#### `src/gemini/response_validator.py`
- **Purpose:** Validates Gemini-returned JSON against strict per-intent schemas. Checks required/allowed keys, types, non-empty strings, positive delay_seconds.
- **Key Class:** `ResponseValidator` — validate_intent_json(data) — returns True/False
- **Status:** ✅ Working — Note: creates AIRegistry() instance on each call (see Issues section)

---

#### `src/gemini/json_parser.py`
- **Purpose:** Strips markdown code fences from Gemini raw output; parses JSON safely.
- **Key Class:** `JsonParser` — parse_gemini_json(raw_text)
- **Status:** ✅ Working

---

#### `src/gemini/gemini_logger.py`
- **Purpose:** Structured API event logging — records model name, latency, inference time, intent, source (GEMINI/LOCAL), retry count, validation result.
- **Key Function:** log_gemini_event(...) — writes to Morphix.GeminiLog logger
- **Status:** ✅ Working

---

### src/intent/ — Local Intent Classification

---

#### `src/intent/intent_classifier.py`
- **Purpose:** Plugin-based offline regex intent classifier. No ML models. Fast, zero-latency.
- **Key Classes:**
  - IntentType (Enum) — ADD_TODO, SHOW_TODO, DELETE_TODO, COMPLETE_TODO, CLEAR_TODO, ADD_REMINDER, SHOW_REMINDERS, DELETE_REMINDER, UNKNOWN
  - IntentResult (dataclass) — intent, parameters, confidence, raw_text
  - TodoIntentPlugin — Regex patterns for all todo intents
  - ReminderIntentPlugin — Regex patterns for all reminder intents
  - IntentClassifier — Plugin registry; classify(text) evaluates all registered plugins
- **Status:** ✅ Working (Layer 4 fallback in the NLU pipeline)

---

#### `src/intent/reminder_parser.py`
- **Purpose:** Rule-based parser for reminder intents. Extracts message, time_expr, and delay (in seconds) from speech.
- **Key Class:** `ReminderIntentParser` — parse(text) — returns dict with intent, message, time_expr, delay
- **Status:** ✅ Working (Layer 2 fallback in the NLU pipeline)

---

### src/todo/ — To-Do Management

---

#### `src/todo/todo_manager.py`
- **Purpose:** Full CRUD to-do management with JSON persistence at storage/todo.json. Uses normalized exact matching to prevent substring errors.
- **Key Class:** `TodoManager`
  - add_task(task_text) / add_todo() — Creates new task; raises ValueError on duplicate
  - get_pending_tasks(), get_all_tasks()
  - complete_task(query) / complete_todo() — Matches by UUID then by normalized text
  - edit_task(query, new_text) / edit_todo() — Updates task text + updated_at
  - delete_task(query) / delete_todo() — Removes matching task
  - clear_all_tasks() — Wipes all tasks
- **Status:** ✅ Working

---

#### `src/todo/todo_models.py`
- **Purpose:** TodoItem dataclass model with to_dict() serializer.
- **Fields:** id, task, created_at, completed, completed_at, updated_at
- **Status:** ✅ Working

---

#### `src/todo/todo_parser.py`
- **Purpose:** Rule-based parser extracting to-do intents and item names from cleaned transcripts.
- **Key Class:** `TodoParser` — parse(clean_text) — returns structured dict
- **Status:** ✅ Working (Layer 3 fallback in the NLU pipeline)

---

### src/reminder/ — Reminder Management

---

#### `src/reminder/reminder_manager.py`
- **Purpose:** Timezone-aware reminder CRUD with JSON persistence at storage/reminders.json. Converts time expressions to datetime objects.
- **Key Functions/Classes:**
  - get_local_timezone() — Reads OS Windows Registry or fallback timezone mappings
  - parse_time_expression(time_expr, tz, now) — Handles relative (in 10 minutes), absolute (5 PM), and weekday (next Monday)
  - ReminderManager
    - add_reminder(message, time_expr) — Saves reminder with trigger_time, status, id
    - add_reminder_by_delay(message, delay_seconds) — Saves reminder from integer delay
    - get_due_reminders(reference_time) — Returns pending reminders past trigger_time
    - mark_completed(reminder_id), cancel_reminder(reminder_id), delete_reminder(query)
    - get_pending_reminders(), get_all_reminders()
- **Dependencies:** zoneinfo, uuid, json, winreg (Windows)
- **Status:** ✅ Working

---

#### `src/reminder/reminder_checker.py`
- **Purpose:** Background daemon thread polling for due reminders every check_interval seconds and triggering the NotificationManager.
- **Key Class:** `ReminderChecker`
  - start(), stop() — Thread lifecycle management
  - check_once() — Single pass: queries due reminders, triggers notification, marks completed
- **Status:** ✅ Working

---

### src/audio/ and src/notification/

---

#### `src/audio/sound_player.py`
- **Purpose:** Cross-platform WAV audio playback using pygame.mixer. Blocking playback. Graceful handling of missing files.
- **Key Class:** `SoundPlayer` — play_sound(sound_file) — returns True/False
- **Status:** ✅ Working

---

#### `src/notification/notification_manager.py`
- **Purpose:** Pluggable notification dispatching. Plugin channel: ConsoleNotificationSender (Bark → TTS → Console print).
- **Key Classes:**
  - NotificationSender (ABC) — send_notification(message, trigger_time_str, current_time_str)
  - ConsoleNotificationSender — Executes: play bark → tts.speak() → print()
  - NotificationManager — Holds registered senders; calls notify()
- **Status:** ✅ Working

---

### src/nlp/, src/database/, src/ai/, src/emotions/

---

#### `src/nlp/transcript_cleaner.py`
- **Purpose:** Normalizes raw Whisper transcripts: lowercases, strips punctuation, translates number words (fifteen → 15), normalizes "to do" → "todo".
- **Key Class:** `TranscriptCleaner` — clean(text) classmethod
- **Status:** ✅ Working

---

#### `src/database/memory.py`
- **Purpose:** Simple JSON-backed persistent key-value memory store.
- **Key Class:** `Memory` — save(), load(), update(), delete(), exists(), clear(), get_all()
- **Status:** ✅ Working — Not yet integrated into the main pipeline

---

#### `src/ai/ai_registry.py`
- **Purpose:** Thread-safe singleton registry for AI providers using lazy initialization.
- **Key Class:** `AIRegistry` (Singleton)
  - register_provider(name, factory_func) — Stores factory callable
  - get_provider(name) — Lazy-loads and caches provider instance
  - get_supported_intents() — Returns registered intent set
- **Status:** ✅ Working

---

#### `src/emotions/expression_manager.py`
- **Purpose:** Manages Morphix facial expression states and emojis (Neutral, Happy, Sad, Angry, Thinking, Sleeping, Listening, Excited, Confused, Protective). No display hardware coupling.
- **Key Class:** `ExpressionManager` — set_expression(), get_current_expression(), get_emoji(), get_description()
- **Status:** ✅ Working — Not yet integrated into the main pipeline

---

## 🎙️ 3. VOICE & STT/TTS PIPELINE IMPLEMENTATION

### Wake Word Detection Pipeline

```
Microphone (16kHz, Mono)
         |
    sounddevice.InputStream
         |
    ContinuousMicrophoneStream._audio_callback()
    |-- ring_buffer.append(data)          <-- 500ms pre-roll buffer (deque)
    +-- recording_queue.put(data)         <-- active recording queue
         |
    record_chunk_with_preroll(duration=3.0s)
    |-- Prepend 500ms ring buffer
    +-- Returns WAV bytes
         |
    WakeWordManager.detect(wav_bytes)
    |-- [PRIMARY]   OpenWakeWordEngine.is_detected()
    |   +-- morphix.tflite / morphix.onnx (custom model - not yet trained)
    |        +-- Falls back to:
    +-- [FALLBACK] FallbackWakeWordEngine.is_detected()
        |-- SpeechToText.transcribe(wav_bytes)  <-- Whisper.cpp
        +-- WakeWordDetector.detect(transcript)
            |-- STEP 1: Primary phrase substring match ("hey morphix" in text)
            |-- STEP 2: Alias substring match (Config.ACCEPTED_ALIASES)
            +-- STEP 3: Fuzzy sliding window matching (RapidFuzz / difflib)
                        Threshold: 0.62 (Config.SIMILARITY_THRESHOLD)
```

**Wake Phrases configured (Config.WAKE_PHRASES):**
- "hey morphix", "hello morphix", "hi morphix", "morphix", "wake up morphix"
- "hey morfix", "hello morfix", "morphics", "hey morphics", "he might fix"
- "hey more fix", "hey more pics", "more fix"

**Fuzzy Algorithm Features:**
- Token-level weighted similarity with filler word dampening ("hey", "hi" weighted 0.5)
- Non-filler similarity scoring to handle phonetic noise
- Sliding window search across transcript tokens
- Protection against single-character and single-token false positives

---

### STT (Whisper.cpp) Pipeline

```
WAV bytes (from ContinuousMicrophoneStream)
         |
SpeechToText.transcribe(wav_bytes)
         |
WhisperEngine.transcribe(audio_source)
|-- Write bytes to temp file: temp/audio/*.wav
|-- WhisperEngine._model.transcribe(audio_path, n_processors=4)
|   +-- pywhispercpp.Model (ggml-tiny.en.bin)
|        [C++ stdout suppressed via suppress_stdout_stderr()]
|-- Join segments: " ".join([seg.text for seg in segments])
|-- Clean [BLANK_AUDIO] tokens
|-- Log RTF (Real-Time Factor)
+-- Return final_sentence (str)
         |
TranscriptCleaner.clean(transcript)
|-- lowercase, punctuation to spaces
|-- "to do" to "todo"
|-- number words to digits
+-- normalize whitespace
         |
is_meaningful_speech(cleaned_command)
         |
handle_incoming_command(cleaned_command, ...)
```

---

### TTS (Piper) Pipeline

```
Response string (from NLU layer or router)
         |
TextToSpeech.speak(text)
|-- _validate_environment()
|   |-- piper.exe path check
|   |-- .onnx model path check
|   +-- .json config path check (warning only)
|-- save_to_file(text, temp/audio/speech_<timestamp>.wav)
|   +-- subprocess.run([piper.exe, --model, .onnx, --output_file, .wav])
|       Input: text.encode("utf-8") via stdin
|-- winsound.PlaySound(temp_wav, SND_FILENAME)  [blocking, Windows]
+-- temp_wav.unlink()  [auto-cleanup]
```

**Model auto-discovery order:**
1. models/piper/*.onnx
2. src/models/piper/*.onnx
3. piper/*.onnx
4. Recursive rglob("*.onnx") throughout project root

---

## 🧠 4. INTENT HANDLING, REMINDERS & TODO LOGIC

### NLU Pipeline — 4-Tier Fallback Architecture

```
Cleaned transcript
        |
        v
[LAYER 1] Gemini NLU (Primary — if API key valid)
        |   GeminiIntentEngine.process_transcript(transcript)
        |   |-- Deduplication check (3s window)
        |   |-- Circuit breaker check (429 guard)
        |   |-- genai.Client.generate_content(PROMPT_INTENT_V1 + transcript)
        |   |-- JsonParser.parse_gemini_json(response.text)
        |   |-- ResponseValidator.validate_intent_json(parsed)
        |   +-- Returns: {"intent": ..., "confidence": ..., "task": ...}
        |
        v (if Gemini unavailable or returns UNKNOWN)
[LAYER 2] Local ReminderIntentParser
        |   ReminderIntentParser.parse(cleaned_command)
        |   +-- Returns SET_REMINDER dict or UNKNOWN
        |
        v (if no reminder match)
[LAYER 3] Local TodoParser
        |   TodoParser.parse(cleaned_command)
        |   +-- Returns ADD_TODO / DELETE_TODO / ... or UNKNOWN
        |
        v (if no todo match)
[LAYER 4] IntentClassifier (Plugin-based Regex)
            IntentClassifier.classify(cleaned_command)
            |-- TodoIntentPlugin.match(text)
            +-- ReminderIntentPlugin.match(text)
```

---

### Gemini API Integration Details

| Setting | Value |
|---|---|
| Model | gemini-flash-lite-latest (env-overridable) |
| Intent Temperature | 0.0 (deterministic) |
| Chat Temperature | 0.7 (conversational) |
| Timeout | 8.0 seconds |
| Max Output Tokens | 512 |
| Max Retries | 2 |
| Confidence Threshold | 0.7 |
| Deduplication Window | 3.0 seconds |
| Circuit Breaker Threshold | 3 consecutive 429s |
| Circuit Breaker Cooldown | 60 seconds |
| Backoff Base | 2.0s (exponential + jitter) |
| Backoff Max Cap | 10.0 seconds |
| Chat History Limit | 10 messages (rolling) |

**Supported Intent JSON Formats:**

```
ADD_TODO:       {"intent": "ADD_TODO",      "confidence": 0.98, "task": "string"}
DELETE_TODO:    {"intent": "DELETE_TODO",   "confidence": 0.95, "task": "string"}
EDIT_TODO:      {"intent": "EDIT_TODO",     "confidence": 0.92, "old_item": "...", "new_item": "..."}
COMPLETE_TODO:  {"intent": "COMPLETE_TODO", "confidence": 0.96, "task": "string"}
SHOW_TODO:      {"intent": "SHOW_TODO",     "confidence": 0.99}
SET_REMINDER:   {"intent": "SET_REMINDER",  "confidence": 0.97, "message": "string", "delay_seconds": 600}
DELETE_REMINDER:{"intent": "DELETE_REMINDER","confidence": 0.95, "target": "string"}
SHOW_REMINDERS: {"intent": "SHOW_REMINDERS","confidence": 0.99}
CHAT:           {"intent": "CHAT",          "confidence": 0.95, "query": "string"}
UNKNOWN:        {"intent": "UNKNOWN",       "confidence": 0.1}
```

---

### JSON Storage Schemas

> Note: Morphix currently uses JSON files for persistence (not SQLite). SQLite is planned for a future phase.

**storage/todo.json Schema:**
```json
[
  {
    "id": "uuid-string",
    "task": "Buy groceries",
    "created_at": "2026-08-03T10:00:00",
    "completed": false,
    "completed_at": null,
    "updated_at": null
  }
]
```

**storage/reminders.json Schema:**
```json
[
  {
    "id": "uuid-string",
    "message": "Drink water",
    "text": "Drink water",
    "created_at": "2026-08-03T10:00:00+05:30",
    "trigger_time": "2026-08-03T10:10:00+05:30",
    "timezone": "Asia/Kolkata",
    "status": "pending",
    "completed": false,
    "repeat": "none"
  }
]
```

Reminder Status Lifecycle: pending → completed (via mark_completed()) or cancelled (via cancel_reminder() / delete_reminder())

---

### Reminder Checker — Background Service

```
Thread: ReminderCheckerThread (daemon=True)
        |
every 10 seconds:
        |
ReminderChecker.check_once()
|-- now = datetime.now(tz)
|-- due_list = reminder_manager.get_due_reminders(now)
+-- for each due reminder:
    |-- NotificationManager.notify(message, trigger_time, current_time)
    |   +-- ConsoleNotificationSender.send_notification()
    |       |-- SoundPlayer.play_sound(wake_sound)       <-- Bark
    |       |-- tts.speak("Tanisha, it is time to ...")  <-- Piper TTS
    |       +-- print() console banner
    +-- reminder_manager.mark_completed(rem_id)
```

---

## ❌ 5. CURRENT ISSUES, BUGS & ERRORS

### Issue 1 — response_validator.py: Unnecessary AIRegistry instantiation per call
- **File:** src/gemini/response_validator.py, Line 45
- **Problem:** `valid_intents = AIRegistry().get_supported_intents()` is called on every Gemini response validation. Although AIRegistry is a singleton, the overhead adds up per API call.
- **Severity:** Minor Performance
- **Fix:** Cache the intent set as a class variable or pass it as a parameter.

---

### Issue 2 — run_voice_assistant.py: process_voice_command() uses add_task() not add_todo()
- **File:** run_voice_assistant.py, Line ~587
- **Problem:** The legacy process_voice_command() calls todo_mgr.add_task(task_str). The response message says "Task added." instead of "Todo added." — slight inconsistency in spoken responses.
- **Severity:** Minor UX inconsistency

---

### Issue 3 — reminder_manager.py: get_due_reminders() filter logic bug (CRITICAL)
- **File:** src/reminder/reminder_manager.py, Line ~330
- **Problem:** The filter condition logic is inverted with a confusing double negative:
  `if rem.get("status", "pending") != "pending" and not rem.get("completed", False) == False:`
  This could cause completed reminders to re-trigger under edge conditions.
- **Severity:** CRITICAL Logic Bug
- **Fix:** Replace with:
  `if rem.get("status", "pending") != "pending" or rem.get("completed", False): continue`

---

### Issue 4 — wake_word.py: Threshold scale inconsistency
- **File:** run_voice_assistant.py Lines 793-794 vs wake_word.py Lines 59-61
- **Problem:** main() passes threshold=0.62 (from Config.SIMILARITY_THRESHOLD) to WakeWordDetector, which multiplies by 100 to store as 62.0 internally. Non-obvious scaling that could cause bugs if Config.SIMILARITY_THRESHOLD is changed.
- **Severity:** Minor Code Clarity

---

### Issue 5 — src/intents/reminder_parser.py: Stub placeholder / duplicate directory
- **File:** src/intents/reminder_parser.py
- **Problem:** An empty/stub placeholder duplicating the path of src/intent/reminder_parser.py. The src/intents/ directory creates confusion alongside src/intent/.
- **Severity:** Code Organization — Dead code

---

### Issue 6 — reminder_manager.py: Windows Timezone Mapping is Incomplete (CRITICAL)
- **File:** src/reminder/reminder_manager.py, Lines 43-53
- **Problem:** The Windows Registry to IANA timezone mapping only covers 7 timezones. Other Windows timezone names will fall through and cause ZoneInfoNotFoundError at runtime.
- **Severity:** CRITICAL — Potential Runtime Error for users outside the 7 mapped timezones
- **Fix:** Use tzdata package or extend the mapping table.

---

### Issue 7 — FallbackWakeWordEngine uses different threshold than main pipeline
- **File:** src/wakeword/wakeword_engine.py, Line 274
- **Problem:** FallbackWakeWordEngine creates WakeWordDetector(threshold=0.70) but main pipeline uses 0.62 via Config.SIMILARITY_THRESHOLD. Creates inconsistent detection sensitivity between the two code paths.
- **Severity:** Behavioral Inconsistency

---

### Issue 8 — No atomic write for JSON storage files
- **Files:** src/todo/todo_manager.py, src/reminder/reminder_manager.py
- **Problem:** If a crash occurs during _write_to_file(), the file could be left partially written. No atomic write (write to temp → rename) pattern is implemented.
- **Severity:** Data Integrity Risk under crash scenarios
- **Fix:** Write to a .tmp file first, then use os.replace() atomically.

---

### Issue 9 — personality.py and expression_manager.py not connected to main pipeline
- **Problem:** Both modules are fully implemented but not imported or called in run_voice_assistant.py. Morphix's expressions and personality are not reflected during runtime.
- **Severity:** Feature Gap — Implementation complete, integration pending

---

### Issue 10 — src/ai/personality/emotion.py — Placeholder / duplicate
- **File:** src/ai/personality/emotion.py
- **Problem:** Role overlaps with src/emotions/expression_manager.py. Likely a legacy file causing confusion.
- **Severity:** Code Organization

---

## 💻 6. FULL CODE COPY (CORE SCRIPTS)

### run_voice_assistant.py — NLU Orchestration Core

```python
# State Machine
class AssistantState(Enum):
    INITIALIZING = "INITIALIZING"
    WAIT_WAKE = "WAIT_WAKE"
    ACTIVE_MODE = "ACTIVE_MODE"
    COMMAND_RECEIVED = "COMMAND_RECEIVED"
    SLEEP = "SLEEP"

# NLU 4-Tier Fallback Orchestrator
def handle_incoming_command(cleaned_command, raw_transcript,
                             reminder_mgr, todo_mgr, sound_player,
                             wake_sound, tts, router, ai_registry,
                             intent_classifier):
    # Layer 1: Gemini NLU
    if router and router.is_available():
        gemini_engine = ai_registry.get_provider("Gemini")
        validated_intent = gemini_engine.process_transcript(cleaned_command)
        if validated_intent.get("intent") != "UNKNOWN":
            response = router.route(validated_intent)
            tts.speak(response)
            return

    # Layer 2: Local Reminder Parser
    parsed_reminder = ReminderIntentParser.parse(cleaned_command)
    if parsed_reminder and parsed_reminder.get("intent") == "SET_REMINDER":
        reminder_mgr.add_reminder(...)
        tts.speak("Reminder saved.")
        return

    # Layer 3: Local Todo Parser
    parsed_todo = TodoParser.parse(cleaned_command)
    if parsed_todo and parsed_todo.get("intent") != "UNKNOWN":
        process_todo_command(parsed_todo, todo_mgr, ...)
        return

    # Layer 4: Plugin-based IntentClassifier
    process_voice_command(cleaned_command, intent_classifier, ...)
```

---

### src/voice/text_to_speech.py — Piper TTS Engine

```python
class TextToSpeech:
    def speak(self, text: str) -> bool:
        if not self._validate_environment():
            return False
        temp_wav = self.temp_audio_dir / f"speech_{int(time.time() * 1000)}.wav"
        success = self.save_to_file(text, str(temp_wav))
        if not success:
            return False
        try:
            import winsound
            winsound.PlaySound(str(temp_wav), winsound.SND_FILENAME)
            return True
        finally:
            self._is_speaking = False
            if temp_wav.exists():
                temp_wav.unlink()

    def save_to_file(self, text: str, output_path: str) -> bool:
        cmd = [str(self.piper_exe), "--model", str(self.model_path),
               "--output_file", str(output_path)]
        if self.config_path and self.config_path.exists():
            cmd.extend(["--config", str(self.config_path)])
        subprocess.run(cmd, input=text.strip().encode("utf-8"),
                       capture_output=True, check=True)
        return Path(output_path).exists() and Path(output_path).stat().st_size > 0
```

---

### src/gemini/gemini_intent_engine.py — Gemini NLU Engine

```python
class GeminiIntentEngine:
    def process_transcript(self, transcript: str) -> Dict[str, Any]:
        now = time.time()
        # Deduplication guard
        if self.last_transcript == transcript and (now - self.last_transcript_time) < 3.0:
            return self.last_result
        # Circuit breaker guard
        if self._is_rate_limited():
            return {"intent": "UNKNOWN"}
        for attempt in range(MAX_RETRY_COUNT + 1):
            try:
                response = self.client.client.models.generate_content(
                    model=self.client.model_name,
                    contents=f"Transcript: {transcript}",
                    config=types.GenerateContentConfig(
                        system_instruction=PROMPT_INTENT_V1,
                        temperature=0.0,
                        max_output_tokens=512,
                        response_mime_type="application/json"
                    )
                )
                parsed = JsonParser.parse_gemini_json(response.text)
                if parsed and ResponseValidator.validate_intent_json(parsed):
                    return parsed
            except Exception as e:
                if self._is_429_error(e):
                    # Exponential backoff + circuit breaker
                    pass
        return {"intent": "UNKNOWN"}
```

---

### src/reminder/reminder_manager.py — Reminder CRUD

```python
def parse_time_expression(time_expr, tz=None, now=None) -> datetime:
    # Relative: "in 10 minutes" -> now + timedelta(minutes=10)
    # Absolute: "tomorrow at 5 PM" -> datetime(tomorrow, 17:00, tz)
    # Weekday: "next Monday at 9 AM" -> next Monday datetime
    ...

class ReminderManager:
    def add_reminder(self, message: str, time_expr: str) -> Dict:
        trigger_dt = parse_time_expression(time_expr, self.tz)
        entry = {
            "id": str(uuid.uuid4()),
            "message": message,
            "text": message,
            "trigger_time": trigger_dt.isoformat(),
            "timezone": str(self.tz),
            "status": "pending",
            "completed": False,
            "repeat": "none"
        }
        reminders = self._read_from_file()
        reminders.append(entry)
        self._write_to_file(reminders)
        return entry
```

---

### src/voice/whisper_engine.py — Whisper.cpp STT

```python
class WhisperEngine(SpeechEngine):
    _model = None  # Class-level singleton

    def initialize(self) -> bool:
        if WhisperEngine._model is not None:
            return True
        from pywhispercpp.model import Model
        WhisperEngine._model = Model(
            model=self.model_name,
            models_dir=self.models_dir,
            print_realtime=False,
            print_progress=False
        )
        return True

    def transcribe(self, audio_source) -> str:
        with suppress_stdout_stderr():
            segments = WhisperEngine._model.transcribe(audio_path, n_processors=4)
        raw = " ".join([seg.text.strip() for seg in segments]).strip()
        return re.sub(r"\[BLANK_AUDIO\]", "", raw).strip()
```

---

### src/voice/wake_word.py — Fuzzy Wake Word Detector

```python
class WakeWordDetector:
    def detect(self, text: str) -> bool:
        _, _, _, _, is_wake = self.compute_similarity(text)
        return is_wake

    def compute_similarity(self, text: str):
        norm_input = self.normalize_text(text)
        # Step 1: Primary phrase substring match
        for phrase in self._primary_phrases:
            if phrase in norm_input:
                return 100.0, phrase, phrase, 1.0, True
        # Step 2: Alias substring match
        for phrase in self._fallback_aliases:
            if self.normalize_text(phrase) in norm_input:
                return 100.0, phrase, phrase, 1.0, True
        # Step 3: Fuzzy sliding window match (RapidFuzz)
        best_score = 0.0
        for phrase in self._wake_phrases:
            phrase_tokens = self.normalize_text(phrase).split()
            for window in sliding_windows(input_tokens, len(phrase_tokens)):
                score = max(
                    self._get_weighted_fuzzy_similarity(window, phrase_tokens),
                    self._get_non_filler_fuzzy_similarity(window, phrase_tokens)
                )
                best_score = max(best_score, score)
        is_detected = best_score >= self._threshold
        return best_score, best_phrase, best_phrase, best_score / 100.0, is_detected
```

---

## 📦 Dependencies Reference (requirements.txt)

| Package | Version | Role |
|---|---|---|
| sounddevice | >=0.4.6 | Continuous mic capture (PortAudio) |
| numpy | >=1.22.0 | Audio buffer operations and RMS VAD |
| pygame | >=2.1.0 | WAV sound effect playback (bark) |
| rapidfuzz | >=3.0.0 | High-performance fuzzy string matching |
| openwakeword | >=0.5.1 | Custom TFLite/ONNX wake word model |
| tflite-runtime | >=2.11.0 | TFLite model inference (OWW) |
| onnxruntime | >=1.12.0 | ONNX model inference (OWW fallback) |
| pywhispercpp | >=0.1.0 | Python bindings for Whisper.cpp |
| google-genai | >=0.1.1 | Official Google Gemini API SDK |
| python-dotenv | >=0.19.0 | .env file loading |

System-level (pre-installed):
- piper.exe + DLLs (in piper/) — Offline neural TTS binary
- ggml-tiny.en.bin (in models/whisper/) — Whisper.cpp STT model

---

*End of PROJECT_SUMMARY.md — Generated by Antigravity AI Architecture Review*

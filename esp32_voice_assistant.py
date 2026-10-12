"""
Morphix ESP32 AI Voice Assistant Server
=======================================
Architecture:
  - Network I/O: Asyncio WebSocket server running on port 8765.
  - Audio Input: Captures incoming 16kHz 16-bit PCM binary stream from ESP32 (INMP441 mic).
  - VAD & STT: Energy-based Voice Activity Detection + Google Speech Recognition.
  - LLM Processing: Grok / Groq API via requests (with context retention and graceful fallbacks).
  - Voice Output: Microsoft Edge TTS (edge-tts) -> 16kHz 16-bit PCM via miniaudio -> WebSocket binary stream to ESP32 (MAX98357A speaker).
  - Display Sync: Whenever Pygame facial expression changes, sends text commands
    ("EXP:LISTEN", "EXP:THINK", "EXP:TALK", "EXP:IDLE") to ESP32 for hardware display sync (ST7789).
  - Laptop Monitor: Non-blocking Pygame GUI running concurrently on the main thread
    displaying animated robot eyes, expressions, audio level meters, and debug status.
"""

from __future__ import annotations

import asyncio
import collections
import io
import json
import logging
import math
import os
import queue
import random
import ssl
import sys
import threading
import time
import wave
from typing import Any, Dict, List, Optional, Set

# External dependencies
import miniaudio
import numpy as np
import pygame
import requests
import speech_recognition as sr
import edge_tts
import websockets

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

# Patch SSL for edge-tts on Windows if needed
try:
    import edge_tts.communicate as _edge_comm
    _patched_ssl_ctx = ssl.create_default_context()
    _patched_ssl_ctx.check_hostname = False
    _patched_ssl_ctx.verify_mode = ssl.CERT_NONE
    _edge_comm._SSL_CTX = _patched_ssl_ctx
except Exception:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [ESP32-Assistant]: %(message)s"
)
logger = logging.getLogger("ESP32Assistant")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8765

# Audio specs
SAMPLE_RATE = 16000           # 16 kHz
SAMPLE_WIDTH = 2              # 16-bit = 2 bytes per sample
CHANNELS = 1                  # Mono
BYTES_PER_SAMPLE = SAMPLE_WIDTH * CHANNELS
BYTES_PER_SECOND = SAMPLE_RATE * BYTES_PER_SAMPLE

# Voice Activity Detection (VAD) parameters
ENERGY_THRESHOLD = 650        # RMS threshold to detect voice from INMP441
SILENCE_TIMEOUT = 1.3         # Seconds of silence after speech to trigger end-of-utterance
MIN_SPEECH_DURATION = 0.5     # Minimum seconds of audio to consider valid speech
MAX_SPEECH_DURATION = 10.0    # Maximum seconds before cutting off utterance
POST_SPEECH_PAUSE = 0.2       # Seconds buffer before switching to thinking

# Microsoft Edge TTS Settings
TTS_VOICE = "en-IN-NeerjaNeural"  # Natural conversational voice (or "en-US-AriaNeural")
TTS_RATE = "+5%"
TTS_PITCH = "+2Hz"
TTS_STREAM_CHUNK_SIZE = 1024      # Chunk size when streaming PCM back to ESP32 (bytes)

# Grok / Groq LLM API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROK_API_KEY = os.getenv("GROK_API_KEY", "") or os.getenv("XAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Pygame Window Configuration
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 400
FPS = 60

# Palette
C_BG = (15, 17, 23)
C_PRIMARY = (0, 229, 255)       # Neon Cyan
C_HIGHLIGHT = (200, 255, 255)   # Ice White
C_ACCENT = (255, 100, 180)      # Neon Pink
C_TEXT = (220, 230, 242)
C_DIM = (70, 85, 105)
C_GREEN = (46, 213, 115)
C_YELLOW = (255, 217, 61)
C_RED = (255, 71, 87)


# ==============================================================================
# SHARED APPLICATION STATE
# ==============================================================================
class AssistantState:
    def __init__(self):
        self.expression: str = "IDLE"     # "IDLE", "LISTEN", "THINK", "TALK"
        self.client_connected: bool = False
        self.client_ip: str = "None"
        self.last_transcript: str = "(Waiting for speech...)"
        self.last_response: str = "Morphix ready. Speak via ESP32 microphone."
        self.mic_level: float = 0.0       # Normalized 0.0 - 1.0 for VU meter
        self.is_processing: bool = False
        self.is_speaking: bool = False
        self.status_msg: str = "Server starting..."
        self.lock = threading.Lock()

    def set_expression(self, expr: str):
        with self.lock:
            self.expression = expr

    def get_expression(self) -> str:
        with self.lock:
            return self.expression


state = AssistantState()

# WebSocket client registry
connected_clients: Set[websockets.WebSocketServerProtocol] = set()
async_loop: Optional[asyncio.AbstractEventLoop] = None


# ==============================================================================
# WEBSOCKET BROADCAST HELPER
# ==============================================================================
async def broadcast_expression(expr: str):
    """Broadcasts expression commands to all connected ESP32 clients (e.g. 'EXP:LISTEN')."""
    state.set_expression(expr)
    command = f"EXP:{expr}"
    logger.info("Syncing Expression -> ESP32: %s", command)

    if not connected_clients:
        return

    dead_clients = set()
    for ws in list(connected_clients):
        try:
            await ws.send(command)
        except Exception as e:
            logger.warning("Failed to send expression to client: %s", e)
            dead_clients.add(ws)

    for dead in dead_clients:
        connected_clients.discard(dead)


def sync_expression_threadsafe(expr: str):
    """Thread-safe expression updater callable from Pygame or worker threads."""
    state.set_expression(expr)
    if async_loop and async_loop.is_running():
        asyncio.run_coroutine_threadsafe(broadcast_expression(expr), async_loop)


# ==============================================================================
# LLM BRAIN (GROK / GROQ API WITH RETRIES AND CONTEXT)
# ==============================================================================
class LLMBrain:
    """Handles conversation intelligence via Grok API or Groq API."""
    def __init__(self):
        self.history: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are Morphix, an intelligent and friendly AI companion robot connected to an ESP32 hardware unit. "
                    "Keep your responses concise, warm, conversational, and direct (1 to 2 short sentences), "
                    "as your words are being read out loud through an ESP32 speaker."
                )
            }
        ]

    def ask(self, user_text: str) -> str:
        self.history.append({"role": "user", "content": user_text})

        # Try xAI Grok first if GROK_API_KEY is configured
        if GROK_API_KEY:
            try:
                logger.info("Querying xAI Grok API...")
                url = "https://api.x.ai/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {GROK_API_KEY}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "grok-beta",
                    "messages": self.history[-6:],
                    "max_tokens": 120,
                    "temperature": 0.7
                }
                resp = requests.post(url, json=payload, headers=headers, timeout=12)
                if resp.status_code == 200:
                    reply = resp.json()["choices"][0]["message"]["content"].strip()
                    self.history.append({"role": "assistant", "content": reply})
                    return reply
                else:
                    logger.warning("Grok API returned %d: %s", resp.status_code, resp.text)
            except Exception as e:
                logger.warning("xAI Grok query error: %s", e)

        # Try Groq Cloud API if GROQ_API_KEY is configured
        if GROQ_API_KEY:
            for model in ["qwen/qwen3.8-27b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]:
                try:
                    logger.info("Querying Groq Cloud API with model '%s'...", model)
                    url = "https://api.groq.com/openai/v1/chat/completions"
                    headers = {
                        "Authorization": f"Bearer {GROQ_API_KEY}",
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "model": model,
                        "messages": self.history[-6:],
                        "max_tokens": 120,
                        "temperature": 0.7
                    }
                    resp = requests.post(url, json=payload, headers=headers, timeout=10)
                    if resp.status_code == 200:
                        reply = resp.json()["choices"][0]["message"]["content"].strip()
                        self.history.append({"role": "assistant", "content": reply})
                        return reply
                    elif resp.status_code == 404:
                        continue  # Try fallback model
                    else:
                        logger.warning("Groq API returned %d: %s", resp.status_code, resp.text)
                except Exception as e:
                    logger.warning("Groq query error with %s: %s", model, e)

        # Smart fallback if API keys are missing or offline
        fallback_replies = [
            f"I heard you say: '{user_text}'. My hardware link is active and working great!",
            f"Hello! I received your audio clearly from the ESP32: '{user_text}'.",
            f"Morphix here! Your voice was transcribed as '{user_text}'. All systems operational!"
        ]
        chosen = random.choice(fallback_replies)
        self.history.append({"role": "assistant", "content": chosen})
        return chosen


brain = LLMBrain()


# ==============================================================================
# TEXT-TO-SPEECH -> 16KHZ 16-BIT PCM CONVERSION
# ==============================================================================
async def synthesize_speech_to_pcm(text: str) -> bytes:
    """
    Synthesizes speech using Microsoft Edge TTS and converts
    the audio to raw 16kHz 16-bit mono PCM bytes in memory using miniaudio.
    """
    logger.info("Synthesizing TTS: '%s'", text)
    try:
        communicate = edge_tts.Communicate(text, voice=TTS_VOICE, rate=TTS_RATE, pitch=TTS_PITCH)
        mp3_chunks = []
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio":
                mp3_chunks.append(chunk["data"])

        mp3_bytes = b"".join(mp3_chunks)
        if not mp3_bytes:
            logger.error("Edge TTS generated 0 audio bytes.")
            return b""

        # Decode MP3 directly to 16000Hz 16-bit signed mono PCM
        decoded = miniaudio.decode(
            mp3_bytes,
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=CHANNELS,
            sample_rate=SAMPLE_RATE
        )
        pcm_bytes = decoded.samples.tobytes()
        logger.info("TTS converted to PCM: %d bytes (%.2f seconds)",
                    len(pcm_bytes), len(pcm_bytes) / BYTES_PER_SECOND)
        return pcm_bytes
    except Exception as e:
        logger.error("TTS conversion to PCM failed: %s", e)
        return b""


# ==============================================================================
# SPEECH RECOGNITION (GOOGLE SPEECH API)
# ==============================================================================
recognizer = sr.Recognizer()
recognizer.energy_threshold = 300
recognizer.dynamic_energy_threshold = False


def transcribe_pcm_with_google(pcm_data: bytes) -> Optional[str]:
    """Transcribes raw 16kHz 16-bit PCM bytes using Google Speech Recognition."""
    try:
        audio = sr.AudioData(pcm_data, sample_rate=SAMPLE_RATE, sample_width=SAMPLE_WIDTH)
        # Attempt primary recognition in English
        try:
            transcript = recognizer.recognize_google(audio, language="en-IN")
            return transcript
        except sr.UnknownValueError:
            # Fallback to en-US
            try:
                return recognizer.recognize_google(audio, language="en-US")
            except Exception:
                return None
    except Exception as e:
        logger.warning("Google STT error: %s", e)
        return None


# ==============================================================================
# FULL CONVERSATION PIPELINE (ASYNC WORKER)
# ==============================================================================
async def process_voice_turn(raw_pcm: bytes, websocket: websockets.WebSocketServerProtocol):
    """
    Processes an end-of-speech utterance:
      1. EXP:THINK -> ESP32 & Pygame
      2. STT via Google Speech
      3. LLM via Grok / Groq
      4. EXP:TALK -> ESP32 & Pygame
      5. TTS -> 16kHz PCM stream to ESP32
      6. EXP:IDLE -> ESP32 & Pygame
    """
    state.is_processing = True

    # 1. Transition to THINKING
    await broadcast_expression("THINK")
    state.status_msg = "Transcribing speech..."

    # 2. Transcribe in worker thread (non-blocking)
    transcript = await asyncio.to_thread(transcribe_pcm_with_google, raw_pcm)

    if not transcript or not transcript.strip():
        logger.info("No clear speech detected.")
        state.status_msg = "Could not understand audio"
        await asyncio.sleep(0.5)
        await broadcast_expression("IDLE")
        state.is_processing = False
        return

    logger.info("Transcript: '%s'", transcript)
    state.last_transcript = transcript
    state.status_msg = f"Heard: '{transcript}' - Generating AI reply..."

    # 3. Query LLM (in worker thread)
    ai_reply = await asyncio.to_thread(brain.ask, transcript)
    logger.info("AI Reply: '%s'", ai_reply)
    state.last_response = ai_reply
    state.status_msg = "Synthesizing voice response..."

    # 4. Synthesize Edge TTS -> 16kHz 16-bit PCM
    pcm_audio = await synthesize_speech_to_pcm(ai_reply)

    if not pcm_audio:
        state.status_msg = "TTS generation error"
        await broadcast_expression("IDLE")
        state.is_processing = False
        return

    # 5. Transition to TALKING
    state.is_speaking = True
    await broadcast_expression("TALK")
    state.status_msg = "Streaming audio to ESP32 speaker..."

    # 6. Stream PCM audio back to ESP32 over WebSocket
    try:
        chunk_size = TTS_STREAM_CHUNK_SIZE
        total_bytes = len(pcm_audio)
        bytes_sent = 0

        # Calculate paced delay: chunk_size / BYTES_PER_SECOND
        # e.g. 1024 / 32000 = ~0.032s. Use slight speedup (0.025s) to stay ahead of buffer
        pace_delay = (chunk_size / BYTES_PER_SECOND) * 0.85

        for offset in range(0, total_bytes, chunk_size):
            if not state.client_connected:
                break
            chunk = pcm_audio[offset : offset + chunk_size]
            await websocket.send(chunk)
            bytes_sent += len(chunk)
            await asyncio.sleep(pace_delay)

        # Allow extra time for ESP32 I2S DMA buffer to finish physical playback
        drain_delay = min(2.0, (len(pcm_audio) / BYTES_PER_SECOND) * 0.2 + 0.5)
        await asyncio.sleep(drain_delay)

    except Exception as e:
        logger.warning("Streaming audio to ESP32 interrupted: %s", e)

    # 7. Return to IDLE
    state.is_speaking = False
    state.is_processing = False
    state.status_msg = "Ready for next voice input"
    await broadcast_expression("IDLE")


# ==============================================================================
# WEBSOCKET SERVER HANDLER
# ==============================================================================
async def esp32_ws_handler(websocket: websockets.WebSocketServerProtocol, *args):
    """
    Handles WebSocket connection from ESP32:
      - Reads incoming binary 16kHz audio stream from INMP441 microphone.
      - Implements Voice Activity Detection (VAD).
      - Sends expression string commands & TTS PCM audio back to ESP32.
    """
    client_ip = websocket.remote_address[0] if websocket.remote_address else "Unknown"
    logger.info("ESP32 Connected from %s", client_ip)
    connected_clients.add(websocket)

    state.client_connected = True
    state.client_ip = f"{client_ip}:{websocket.remote_address[1]}"
    state.status_msg = f"ESP32 Connected ({state.client_ip})"

    # Sync initial expression
    await websocket.send(f"EXP:{state.get_expression()}")

    # VAD state tracking for this connection
    audio_buffer = bytearray()
    is_speaking_utterance = False
    last_voice_time = time.time()
    utterance_start_time = 0.0

    try:
        async for message in websocket:
            # Handle binary audio frame from ESP32 INMP441
            if isinstance(message, bytes):
                # If Morphix is currently speaking or processing, skip incoming audio to avoid echo
                if state.is_speaking or state.is_processing:
                    continue

                # Process 16-bit PCM chunk
                samples = np.frombuffer(message, dtype=np.int16)
                if len(samples) == 0:
                    continue

                # Compute RMS energy for VAD & display meter
                rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
                state.mic_level = min(1.0, rms / 3000.0)

                current_time = time.time()

                if rms > ENERGY_THRESHOLD:
                    # Voice energy detected
                    if not is_speaking_utterance:
                        # Speech started!
                        is_speaking_utterance = True
                        utterance_start_time = current_time
                        audio_buffer.clear()
                        state.status_msg = "Listening to voice input..."
                        await broadcast_expression("LISTEN")

                    audio_buffer.extend(message)
                    last_voice_time = current_time

                elif is_speaking_utterance:
                    # Silence detected while already recording an utterance
                    audio_buffer.extend(message)

                    # Check if silence threshold or max duration reached
                    silence_elapsed = current_time - last_voice_time
                    utterance_duration = current_time - utterance_start_time

                    if silence_elapsed >= SILENCE_TIMEOUT or utterance_duration >= MAX_SPEECH_DURATION:
                        # Speech finished!
                        is_speaking_utterance = False
                        captured_audio = bytes(audio_buffer)
                        audio_buffer.clear()

                        duration_sec = len(captured_audio) / BYTES_PER_SECOND
                        if duration_sec >= MIN_SPEECH_DURATION:
                            logger.info("Captured utterance: %.2fs (%d bytes). Processing...",
                                        duration_sec, len(captured_audio))
                            # Dispatch processing asynchronously so WS loop continues reading
                            asyncio.create_task(process_voice_turn(captured_audio, websocket))
                        else:
                            logger.debug("Utterance too short (%.2fs), ignored.", duration_sec)
                            await broadcast_expression("IDLE")

            elif isinstance(message, str):
                # Text commands from ESP32 (e.g. touch button, telemetry, ping)
                text = message.strip()
                logger.info("Received text from ESP32: %s", text)
                if text == "PING":
                    await websocket.send("PONG")
                elif text.startswith("TOUCH"):
                    # Touch sensor pressed on ESP32
                    logger.info("Hardware touch triggered on ESP32!")

    except websockets.exceptions.ConnectionClosed:
        logger.info("ESP32 disconnected (%s)", client_ip)
    except Exception as e:
        logger.warning("WebSocket handler exception: %s", e)
    finally:
        connected_clients.discard(websocket)
        if not connected_clients:
            state.client_connected = False
            state.client_ip = "None"
            state.status_msg = "Waiting for ESP32 connection..."
            state.mic_level = 0.0


async def start_websocket_server():
    """Runs the asyncio WebSocket server."""
    global async_loop
    async_loop = asyncio.get_running_loop()
    logger.info("Starting WebSocket server on ws://%s:%d ...", SERVER_HOST, SERVER_PORT)
    state.status_msg = f"Server listening on ws://{SERVER_HOST}:{SERVER_PORT}"

    async with websockets.serve(esp32_ws_handler, SERVER_HOST, SERVER_PORT):
        await asyncio.Future()  # Run forever


def run_asyncio_thread():
    """Entry point for the asyncio server thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(start_websocket_server())
    except Exception as e:
        logger.error("Asyncio loop stopped: %s", e)


# ==============================================================================
# PYGAME EXPRESSION MONITOR (RUNS ON MAIN THREAD)
# ==============================================================================
class EyeRenderer:
    """Renders animated expressive eyes with glowing effects."""
    def __init__(self):
        self.blink_progress = 0.0
        self.is_blinking = False
        self.last_blink_time = time.time()
        self.blink_interval = 3.5
        self.talk_phase = 0.0
        self.think_angle = 0.0

    def update(self, dt: float, expression: str):
        now = time.time()
        # Random blinking in IDLE and LISTEN
        if expression in ("IDLE", "LISTEN"):
            if not self.is_blinking and (now - self.last_blink_time > self.blink_interval):
                self.is_blinking = True
                self.blink_progress = 0.0
                self.blink_interval = random.uniform(2.5, 5.0)

            if self.is_blinking:
                self.blink_progress += dt * 8.0
                if self.blink_progress >= math.pi:
                    self.is_blinking = False
                    self.last_blink_time = now

        # Talk mouth/eye bounce
        if expression == "TALK":
            self.talk_phase += dt * 10.0

        # Think swirl
        if expression == "THINK":
            self.think_angle += dt * 4.0

    def draw(self, surface: pygame.Surface, expression: str, mic_level: float):
        cx, cy = WINDOW_WIDTH // 2, 130
        eye_spacing = 75
        left_cx = cx - eye_spacing
        right_cx = cx + eye_spacing

        blink_scale = 1.0
        if self.is_blinking:
            blink_scale = max(0.08, abs(math.cos(self.blink_progress)))

        if expression == "IDLE":
            # Friendly rounded pill eyes with gentle breathing
            w, h = 48, int(72 * blink_scale)
            color = C_PRIMARY
            self._draw_pill(surface, left_cx, cy, w, h, color)
            self._draw_pill(surface, right_cx, cy, w, h, color)

            # Subtle smile mouth
            pygame.draw.arc(surface, C_PRIMARY, (cx - 24, cy + 50, 48, 20), math.pi * 0.15, math.pi * 0.85, 3)

        elif expression == "LISTEN":
            # Wide, alert, listening eyes pulsing with mic level
            pulse = 1.0 + (mic_level * 0.35)
            w, h = int(54 * pulse), int(80 * blink_scale * pulse)
            color = C_YELLOW
            self._draw_pill(surface, left_cx, cy, w, h, color)
            self._draw_pill(surface, right_cx, cy, w, h, color)

            # Listening sound bars under eyes
            bar_w = 40
            bar_h = int(6 + mic_level * 24)
            pygame.draw.rect(surface, color, (left_cx - bar_w // 2, cy + 60, bar_w, bar_h), border_radius=4)
            pygame.draw.rect(surface, color, (right_cx - bar_w // 2, cy + 60, bar_w, bar_h), border_radius=4)

        elif expression == "THINK":
            # Looking upward with revolving thinking particles
            w, h = 46, 52
            color = C_ACCENT
            self._draw_pill(surface, left_cx, cy - 8, w, h, color)
            self._draw_pill(surface, right_cx, cy - 8, w, h, color)

            # Swirling orbit particles around head
            for i in range(4):
                angle = self.think_angle + (i * (math.pi / 2))
                px = cx + int(math.cos(angle) * 110)
                py = cy + int(math.sin(angle) * 35) - 10
                pygame.draw.circle(surface, color, (px, py), 4)

            # Small "o" curious mouth
            pygame.draw.circle(surface, color, (cx, cy + 52), 6, 2)

        elif expression == "TALK":
            # Speaking animated eyes + moving mouth
            mouth_bounce = int(abs(math.sin(self.talk_phase)) * 16)
            w, h = 50, 70
            color = C_GREEN
            self._draw_pill(surface, left_cx, cy, w, h, color)
            self._draw_pill(surface, right_cx, cy, w, h, color)

            # Animated expressive mouth
            mouth_rect = pygame.Rect(cx - 22, cy + 46, 44, 10 + mouth_bounce)
            pygame.draw.ellipse(surface, color, mouth_rect)

    def _draw_pill(self, surface: pygame.Surface, cx: int, cy: int, w: int, h: int, color: tuple):
        rect = pygame.Rect(cx - w // 2, cy - h // 2, w, max(4, h))
        # Draw soft outer glow
        glow_rect = rect.inflate(12, 12)
        glow_surf = pygame.Surface((glow_rect.w, glow_rect.h), pygame.SRCALPHA)
        glow_color = (*color[:3], 45)
        pygame.draw.rect(glow_surf, glow_color, glow_surf.get_rect(), border_radius=int(w * 0.45))
        surface.blit(glow_surf, glow_rect.topleft)

        # Draw main eye pill
        pygame.draw.rect(surface, color, rect, border_radius=int(w * 0.42))
        # Inner white specular highlight
        if h > 20:
            hi_rect = pygame.Rect(cx - w // 4, cy - h // 3, w // 3, h // 4)
            pygame.draw.ellipse(surface, C_HIGHLIGHT, hi_rect)


def run_pygame_monitor():
    """Main Pygame application loop."""
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Morphix AI — ESP32 Voice & Expression Bridge")
    clock = pygame.time.Clock()

    font_title = pygame.font.SysFont("segoeui,consolas,arial", 17, bold=True)
    font_main = pygame.font.SysFont("segoeui,consolas,arial", 13)
    font_bold = pygame.font.SysFont("segoeui,consolas,arial", 13, bold=True)
    font_tiny = pygame.font.SysFont("consolas,arial", 11)

    eye_renderer = EyeRenderer()
    running = True
    prev_time = time.time()

    print("\n" + "=" * 60)
    print("      MORPHIX ESP32 WEBSOCKET ASSISTANT RUNNING")
    print("=" * 60)
    print(f"  WebSocket URL : ws://{SERVER_HOST}:{SERVER_PORT}")
    print(f"  Pygame Monitor: {WINDOW_WIDTH}x{WINDOW_HEIGHT} Active")
    print("  Controls:")
    print("    [1] Set IDLE    [2] Set LISTEN    [3] Set THINK    [4] Set TALK")
    print("    [ESC] or Close Window to exit")
    print("=" * 60 + "\n")

    while running:
        now = time.time()
        dt = now - prev_time
        prev_time = now

        # Handle Pygame window events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_1:
                    sync_expression_threadsafe("IDLE")
                elif event.key == pygame.K_2:
                    sync_expression_threadsafe("LISTEN")
                elif event.key == pygame.K_3:
                    sync_expression_threadsafe("THINK")
                elif event.key == pygame.K_4:
                    sync_expression_threadsafe("TALK")

        # Update animated facial expressions
        current_expr = state.get_expression()
        eye_renderer.update(dt, current_expr)

        # Clear background
        screen.fill(C_BG)

        # Draw Face Canvas Area
        face_rect = pygame.Rect(16, 16, WINDOW_WIDTH - 32, 220)
        pygame.draw.rect(screen, (22, 26, 36), face_rect, border_radius=12)
        pygame.draw.rect(screen, (38, 45, 62), face_rect, width=1, border_radius=12)

        # Render Expressive Eyes
        eye_renderer.draw(screen, current_expr, state.mic_level)

        # Render Expression Tag Badge
        expr_colors = {
            "IDLE": C_PRIMARY,
            "LISTEN": C_YELLOW,
            "THINK": C_ACCENT,
            "TALK": C_GREEN
        }
        badge_color = expr_colors.get(current_expr, C_PRIMARY)
        badge_text = font_bold.render(f"EXP:{current_expr}", True, badge_color)
        screen.blit(badge_text, (face_rect.right - badge_text.get_width() - 14, 26))

        # Render Status header
        status_color = C_GREEN if state.client_connected else C_DIM
        status_dot = font_title.render("●", True, status_color)
        screen.blit(status_dot, (30, 24))

        client_str = f"ESP32: {state.client_ip}" if state.client_connected else "ESP32: Disconnected (Listening :8765)"
        title_surf = font_title.render(client_str, True, C_TEXT)
        screen.blit(title_surf, (48, 25))

        # Draw VU Mic Level Bar (if connected)
        meter_x, meter_y, meter_w, meter_h = 30, face_rect.bottom - 22, 220, 8
        pygame.draw.rect(screen, (35, 42, 58), (meter_x, meter_y, meter_w, meter_h), border_radius=4)
        active_w = int(meter_w * min(1.0, state.mic_level))
        if active_w > 0:
            meter_col = C_YELLOW if state.mic_level > 0.2 else C_PRIMARY
            pygame.draw.rect(screen, meter_col, (meter_x, meter_y, active_w, meter_h), border_radius=4)
        meter_label = font_tiny.render("INMP441 MIC RMS", True, C_DIM)
        screen.blit(meter_label, (meter_x + meter_w + 10, meter_y - 2))

        # Bottom Debug & Conversation Card
        card_rect = pygame.Rect(16, 248, WINDOW_WIDTH - 32, 136)
        pygame.draw.rect(screen, (22, 26, 36), card_rect, border_radius=12)
        pygame.draw.rect(screen, (38, 45, 62), card_rect, width=1, border_radius=12)

        # Status text
        stat_text = font_main.render(f"Status: {state.status_msg}", True, C_HIGHLIGHT)
        screen.blit(stat_text, (28, 256))

        # Last User Speech Transcript
        transcript_str = state.last_transcript
        if len(transcript_str) > 65:
            transcript_str = transcript_str[:62] + "..."
        t_label = font_bold.render("User:", True, C_YELLOW)
        t_val = font_main.render(transcript_str, True, C_TEXT)
        screen.blit(t_label, (28, 280))
        screen.blit(t_val, (75, 280))

        # Last Assistant Response
        resp_str = state.last_response
        if len(resp_str) > 75:
            resp_str = resp_str[:72] + "..."
        r_label = font_bold.render("Morphix:", True, C_GREEN)
        r_val = font_main.render(resp_str, True, C_TEXT)
        screen.blit(r_label, (28, 304))
        screen.blit(r_val, (95, 304))

        # Key shortcut guide
        guide_surf = font_tiny.render("Keys: [1] IDLE  [2] LISTEN  [3] THINK  [4] TALK  |  ESP32 ST7789 synced via 'EXP:<NAME>'", True, C_DIM)
        screen.blit(guide_surf, (28, 355))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    print("\nPygame monitor closed. Shutting down assistant...")
    os._exit(0)


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================
def main():
    # 1. Start Asyncio WebSocket server on background daemon thread
    server_thread = threading.Thread(target=run_asyncio_thread, daemon=True, name="WebSocketServerThread")
    server_thread.start()

    # Small delay for socket bind
    time.sleep(0.3)

    # 2. Start Pygame display loop on the MAIN thread (ensures stable OS window pump)
    try:
        run_pygame_monitor()
    except KeyboardInterrupt:
        print("\nExiting via KeyboardInterrupt.")
        sys.exit(0)


if __name__ == "__main__":
    main()

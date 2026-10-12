"""
Morphix Brain - Headless AI Systems Coordinator
================================================
Server: Windows Laptop (Headless terminal server - NO Pygame GUI)
Client: ESP32 over Wi-Fi (ST7789 Display, INMP441 Mic, MAX98357A Speaker)

Workflow:
  Phase 1: Standby & Wake Word
    - ESP32 streams raw 16kHz 16-bit PCM from INMP441 mic.
    - ESP32 display is in standby (EXP:SLEEP, closed eyes).
    - Laptop runs continuous background Google Speech Recognition listening
      specifically for "Wake up Morphix" (and natural variations).

  Phase 2: The Awakening
    - Wake word detected -> Laptop instantly sends "EXP:AWAKE".
    - ESP32 renders open eyes via procedural geometry.
    - Laptop loads "barak.wav" (or fallback), converts to 16kHz PCM, and streams
      it to the ESP32 speaker.

  Phase 3: The Conversation
    - Laptop listens for user's prompt (EXP:LISTEN).
    - VAD detects speech completion -> Laptop sends "EXP:THINK".
    - Prompt sent to Grok / Groq API for response.
    - Response synthesized via Microsoft Edge TTS -> converted to 16kHz PCM.
    - Laptop sends "EXP:TALK" (ESP32 draws speaking mouth).
    - Laptop streams TTS audio chunks over WebSocket to ESP32 speaker.
    - Playback finishes -> Laptop sends "EXP:IDLE".
    - After inactivity timeout -> Laptop sends "EXP:SLEEP" (returns to standby).
"""

from __future__ import annotations

import asyncio
import io
import logging
import math
import os
import random
import re
import ssl
import sys
import time
import wave
from pathlib import Path
from typing import Dict, List, Optional, Set

# External dependencies (NO Pygame imported)
import miniaudio
import numpy as np
import requests
import speech_recognition as sr
import edge_tts
import websockets

# Load .env if present
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

# Terminal ANSI Styling
class LogColors:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    MAGENTA = "\033[95m"
    DIM     = "\033[90m"

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("MorphixBrain")


# ==============================================================================
# CONFIGURATION
# ==============================================================================
HOST = "0.0.0.0"
PORT = 8765

# Audio constants
SAMPLE_RATE = 16000          # 16 kHz
SAMPLE_WIDTH = 2             # 16-bit PCM = 2 bytes
CHANNELS = 1                 # Mono
BYTES_PER_SECOND = SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS

# VAD & Wake Word tuning
VAD_ENERGY_THRESHOLD = 600   # RMS threshold for INMP441 voice detection
SILENCE_TIMEOUT = 1.3        # Seconds of silence after speech to complete utterance
MIN_SPEECH_DURATION = 0.6    # Minimum speech duration to process (seconds)
MAX_SPEECH_DURATION = 10.0   # Maximum speech duration cutoff
CONVERSATION_TIMEOUT = 18.0  # Seconds of inactivity before sleeping again

# TTS settings
TTS_VOICE = "en-IN-NeerjaNeural"
TTS_RATE = "+5%"
TTS_PITCH = "+2Hz"
TTS_CHUNK_SIZE = 1024        # Bytes per WebSocket chunk

# API Keys
GROK_API_KEY = os.getenv("GROK_API_KEY", "") or os.getenv("XAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Wake word variations
WAKE_WORDS = [
    "wake up morphix",
    "wakeup morphix",
    "wake up morfix",
    "wake up morphics",
    "hey morphix",
    "hello morphix",
    "morphix wake up",
    "morphix"
]


# ==============================================================================
# LLM BRAIN (GROK / GROQ API WITH RETENTION)
# ==============================================================================
class GrokAssistant:
    def __init__(self):
        self.history: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are Morphix, a helpful, charming, and expressive companion robot. "
                    "Your responses will be spoken aloud through an ESP32 speaker, "
                    "so keep your answers natural, engaging, and brief (1 to 2 short sentences)."
                )
            }
        ]

    def chat(self, user_prompt: str) -> str:
        self.history.append({"role": "user", "content": user_prompt})

        # 1. Try xAI Grok API
        if GROK_API_KEY:
            try:
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
                res = requests.post(url, json=payload, headers=headers, timeout=12)
                if res.status_code == 200:
                    reply = res.json()["choices"][0]["message"]["content"].strip()
                    self.history.append({"role": "assistant", "content": reply})
                    return reply
                else:
                    logger.warning("Grok API response %d: %s", res.status_code, res.text)
            except Exception as e:
                logger.warning("Grok API error: %s", e)

        # 2. Try Groq Cloud API
        if GROQ_API_KEY:
            for model_id in ["qwen/qwen3.8-27b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]:
                try:
                    url = "https://api.groq.com/openai/v1/chat/completions"
                    headers = {
                        "Authorization": f"Bearer {GROQ_API_KEY}",
                        "Content-Type": "application/json"
                    }
                    payload = {
                        "model": model_id,
                        "messages": self.history[-6:],
                        "max_tokens": 120,
                        "temperature": 0.7
                    }
                    res = requests.post(url, json=payload, headers=headers, timeout=10)
                    if res.status_code == 200:
                        reply = res.json()["choices"][0]["message"]["content"].strip()
                        self.history.append({"role": "assistant", "content": reply})
                        return reply
                    elif res.status_code == 404:
                        continue
                except Exception as e:
                    logger.warning("Groq (%s) error: %s", model_id, e)

        # 3. Fallback response
        fallback = f"I heard: '{user_prompt}'. All Morphix hardware links are active!"
        self.history.append({"role": "assistant", "content": fallback})
        return fallback


brain = GrokAssistant()


# ==============================================================================
# AUDIO SYNTHESIS & FILE LOADER (PCM 16KHZ)
# ==============================================================================
def load_notification_sound() -> bytes:
    """
    Loads 'barak.wav' or fallback sound file, converts it to raw
    16kHz 16-bit signed mono PCM bytes via miniaudio.
    """
    candidates = [
        "barak.wav",
        "assets/sounds/morphix_voice.wav",
        "assets/sounds/bark.wav"
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                decoded = miniaudio.decode_file(
                    path,
                    output_format=miniaudio.SampleFormat.SIGNED16,
                    nchannels=CHANNELS,
                    sample_rate=SAMPLE_RATE
                )
                pcm = decoded.samples.tobytes()
                logger.info("Loaded notification sound '%s' (%d bytes, %.2fs)",
                            path, len(pcm), len(pcm) / BYTES_PER_SECOND)
                return pcm
            except Exception as e:
                logger.warning("Failed decoding '%s': %s", path, e)

    # In-memory synthesized chime if no file exists
    logger.info("Generating in-memory awakening chime...")
    t = np.linspace(0, 0.4, int(SAMPLE_RATE * 0.4), endpoint=False)
    tone = (np.sin(2 * np.pi * 587.33 * t) * 0.5 + np.sin(2 * np.pi * 880.0 * t) * 0.5)
    tone = (tone * 32767 * 0.7).astype(np.int16)
    return tone.tobytes()


async def text_to_16k_pcm(text: str) -> bytes:
    """Synthesizes text using Microsoft Edge TTS and converts to 16kHz mono PCM."""
    try:
        comm = edge_tts.Communicate(text, voice=TTS_VOICE, rate=TTS_RATE, pitch=TTS_PITCH)
        mp3_chunks = []
        async for chunk in comm.stream():
            if chunk.get("type") == "audio":
                mp3_chunks.append(chunk["data"])
        mp3_data = b"".join(mp3_chunks)
        if not mp3_data:
            return b""

        decoded = miniaudio.decode(
            mp3_data,
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=CHANNELS,
            sample_rate=SAMPLE_RATE
        )
        return decoded.samples.tobytes()
    except Exception as e:
        logger.error("TTS synthesis error: %s", e)
        return b""


# ==============================================================================
# SPEECH RECOGNITION (GOOGLE SPEECH API)
# ==============================================================================
recognizer = sr.Recognizer()
recognizer.energy_threshold = 300
recognizer.dynamic_energy_threshold = False


def transcribe_pcm(pcm_bytes: bytes) -> Optional[str]:
    """Transcribes raw 16kHz 16-bit PCM bytes using Google Speech Recognition."""
    try:
        audio = sr.AudioData(pcm_bytes, sample_rate=SAMPLE_RATE, sample_width=SAMPLE_WIDTH)
        try:
            text = recognizer.recognize_google(audio, language="en-IN")
        except sr.UnknownValueError:
            text = recognizer.recognize_google(audio, language="en-US")
        
        print(f"{LogColors.YELLOW}[HEARD FROM MIC]:{LogColors.RESET} \"{text}\"")
        return text
    except Exception:
        return None


def is_wake_word(transcript: str) -> bool:
    """Checks if transcript matches any wake word variations."""
    cleaned = re.sub(r"[^\w\s]", "", transcript.lower()).strip()
    for w in WAKE_WORDS:
        if w in cleaned:
            return True
    return False


# ==============================================================================
# HEADLESS STATE MACHINE & WEBSOCKET SERVER
# ==============================================================================
class MorphixSession:
    """Encapsulates the 3-phase workflow for an ESP32 connection."""

    def __init__(self, websocket: websockets.WebSocketServerProtocol):
        self.ws = websocket
        self.client_ip = websocket.remote_address[0] if websocket.remote_address else "Unknown"
        self.state = "STANDBY"
        self.last_interaction_time = time.time()
        self.notification_audio = load_notification_sound()

    async def send_expression(self, expr_command: str):
        print(f"{LogColors.CYAN}[SYNC]{LogColors.RESET} Sending Display Command -> {LogColors.BOLD}{expr_command}{LogColors.RESET}")
        try:
            await self.ws.send(expr_command)
        except Exception as e:
            logger.warning("Failed sending expression '%s': %s", expr_command, e)

    async def stream_audio_to_esp32(self, pcm_bytes: bytes):
        total = len(pcm_bytes)
        pace = (TTS_CHUNK_SIZE / BYTES_PER_SECOND) * 0.82

        for offset in range(0, total, TTS_CHUNK_SIZE):
            chunk = pcm_bytes[offset : offset + TTS_CHUNK_SIZE]
            await self.ws.send(chunk)
            await asyncio.sleep(pace)

        drain_wait = min(1.5, (total / BYTES_PER_SECOND) * 0.2 + 0.3)
        await asyncio.sleep(drain_wait)

    async def run(self):
        print(f"\n{LogColors.GREEN}======================================================{LogColors.RESET}")
        print(f"{LogColors.GREEN}[CONNECTED]{LogColors.RESET} ESP32 link established from {LogColors.BOLD}{self.client_ip}{LogColors.RESET}")
        print(f"{LogColors.GREEN}======================================================{LogColors.RESET}\n")

        self.state = "STANDBY"
        await self.send_expression("EXP:SLEEP")
        print(f"{LogColors.DIM}[STANDBY] Display: Closed Eyes (EXP:SLEEP). Listening for wake word...{LogColors.RESET}")

        audio_buffer = bytearray()
        rolling_wake_buffer = bytearray()
        is_capturing_speech = False
        speech_start_time = 0.0
        last_speech_energy_time = time.time()

        inactivity_task = asyncio.create_task(self._inactivity_watchdog())

        try:
            async for message in self.ws:
                if not isinstance(message, bytes):
                    continue

                if self.state in ("SPEAKING", "PROCESSING"):
                    continue

                samples = np.frombuffer(message, dtype=np.int16)
                if len(samples) == 0:
                    continue

                rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
                print(f"[MIC RMS LEVEL]: {rms:.1f}", end="\r")

                now = time.time()

                if self.state == "STANDBY":
                    rolling_wake_buffer.extend(message)
                    if len(rolling_wake_buffer) > 96000:
                        del rolling_wake_buffer[:len(rolling_wake_buffer) - 96000]

                    if rms > VAD_ENERGY_THRESHOLD:
                        last_speech_energy_time = now
                        if not is_capturing_speech:
                            is_capturing_speech = True

                    elif is_capturing_speech:
                        if (now - last_speech_energy_time) >= 1.0:
                            is_capturing_speech = False
                            if len(rolling_wake_buffer) >= (BYTES_PER_SECOND * 0.8):
                                raw_slice = bytes(rolling_wake_buffer)
                                transcript = await asyncio.to_thread(transcribe_pcm, raw_slice)
                                if transcript:
                                    if is_wake_word(transcript):
                                        await self._handle_awakening()
                                        rolling_wake_buffer.clear()
                                        audio_buffer.clear()

                elif self.state in ("AWAKE", "LISTENING"):
                    if rms > VAD_ENERGY_THRESHOLD:
                        if not is_capturing_speech:
                            is_capturing_speech = True
                            speech_start_time = now
                            audio_buffer.clear()
                            await self.send_expression("EXP:LISTEN")
                            print(f"{LogColors.CYAN}[MIC ACTIVE]{LogColors.RESET} User is speaking...")

                        audio_buffer.extend(message)
                        last_speech_energy_time = now

                    elif is_capturing_speech:
                        audio_buffer.extend(message)
                        silence_duration = now - last_speech_energy_time
                        utterance_len = now - speech_start_time

                        if silence_duration >= SILENCE_TIMEOUT or utterance_len >= MAX_SPEECH_DURATION:
                            is_capturing_speech = False
                            captured = bytes(audio_buffer)
                            audio_buffer.clear()

                            if len(captured) >= (BYTES_PER_SECOND * MIN_SPEECH_DURATION):
                                await self._process_conversation_prompt(captured)
                            else:
                                await self.send_expression("EXP:IDLE")

        except websockets.exceptions.ConnectionClosed:
            print(f"\n{LogColors.RED}[DISCONNECTED]{LogColors.RESET} ESP32 disconnected.")
        finally:
            inactivity_task.cancel()

    async def _handle_awakening(self):
        print(f"\n{LogColors.GREEN}[WAKE WORD DETECTED!]{LogColors.RESET} -> Awakening Morphix...")
        self.state = "AWAKE"
        self.last_interaction_time = time.time()

        await self.send_expression("EXP:AWAKE")

        if self.notification_audio:
            print(f"{LogColors.MAGENTA}[AUDIO]{LogColors.RESET} Streaming notification chime/bark to ESP32 speaker...")
            self.state = "SPEAKING"
            await self.stream_audio_to_esp32(self.notification_audio)

        self.state = "LISTENING"
        await self.send_expression("EXP:LISTEN")
        print(f"{LogColors.CYAN}[READY]{LogColors.RESET} Listening for user prompt...\n")

    async def _process_conversation_prompt(self, pcm_data: bytes):
        self.state = "PROCESSING"
        self.last_interaction_time = time.time()

        await self.send_expression("EXP:THINK")
        print(f"{LogColors.YELLOW}[THINKING]{LogColors.RESET} Transcribing audio...")
        transcript = await asyncio.to_thread(transcribe_pcm, pcm_data)

        if not transcript or not transcript.strip():
            print(f"{LogColors.DIM}[STT] Could not understand speech.{LogColors.RESET}")
            await self.send_expression("EXP:IDLE")
            self.state = "LISTENING"
            return

        print(f"{LogColors.YELLOW}[GROK]{LogColors.RESET} Requesting AI response...")
        ai_reply = await asyncio.to_thread(brain.chat, transcript)
        print(f"{LogColors.GREEN}[MORPHIX]{LogColors.RESET} \"{ai_reply}\"")

        print(f"{LogColors.MAGENTA}[TTS]{LogColors.RESET} Synthesizing Microsoft TTS voice to 16kHz PCM...")
        tts_pcm = await text_to_16k_pcm(ai_reply)

        if not tts_pcm:
            logger.error("TTS output was empty.")
            await self.send_expression("EXP:IDLE")
            self.state = "LISTENING"
            return

        self.state = "SPEAKING"
        await self.send_expression("EXP:TALK")
        print(f"{LogColors.MAGENTA}[AUDIO]{LogColors.RESET} Streaming TTS reply to ESP32 speaker...")
        await self.stream_audio_to_esp32(tts_pcm)

        self.state = "LISTENING"
        await self.send_expression("EXP:LISTEN")
        print(f"{LogColors.CYAN}[READY]{LogColors.RESET} Listening for next prompt...\n")

    async def _inactivity_watchdog(self):
        while True:
            await asyncio.sleep(2.0)
            if self.state in ("AWAKE", "LISTENING", "SPEAKING"):
                if (time.time() - self.last_interaction_time) >= CONVERSATION_TIMEOUT:
                    print(f"\n{LogColors.DIM}[INACTIVITY TIMEOUT] Returning Morphix to standby sleep...{LogColors.RESET}")
                    self.state = "STANDBY"
                    await self.send_expression("EXP:SLEEP")


# ==============================================================================
# MAIN WEBSOCKET SERVER LOOP
# ==============================================================================
async def main():
    print(f"\n{LogColors.CYAN}{'='*65}{LogColors.RESET}")
    print(f"{LogColors.BOLD}      MORPHIX HEADLESS AI ROBOTICS SERVER (Windows){LogColors.RESET}")
    print(f"{LogColors.CYAN}{'='*65}{LogColors.RESET}")
    print(f"  {LogColors.BOLD}Server Address{LogColors.RESET} : ws://{HOST}:{PORT}")
    print(f"  {LogColors.BOLD}Speech Engine{LogColors.RESET}  : Google Speech Recognition (16kHz PCM)")
    print(f"  {LogColors.BOLD}AI Core{LogColors.RESET}        : Grok / Groq Cloud API")
    print(f"  {LogColors.BOLD}Voice Engine{LogColors.RESET}   : Microsoft Edge TTS ({TTS_VOICE})")
    print(f"  {LogColors.BOLD}GUI Status{LogColors.RESET}     : Headless (NO Pygame GUI)")
    print(f"  {LogColors.BOLD}Wake Phrase{LogColors.RESET}    : 'Wake up Morphix'")
    print(f"{LogColors.CYAN}{'='*65}{LogColors.RESET}\n")

    logger.info("Waiting for ESP32 hardware to connect over Wi-Fi...")

    async def handler(websocket):
        session = MorphixSession(websocket)
        await session.run()

    async with websockets.serve(handler, HOST, PORT, ping_interval=20, ping_timeout=20):
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{LogColors.RED}Server shutdown cleanly via KeyboardInterrupt.{LogColors.RESET}")
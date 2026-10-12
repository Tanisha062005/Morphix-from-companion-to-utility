"""
Morphix Desktop Companion Robot - Groq Conversational Brain
===========================================================
File: src/ai/groq_chat.py
Description:
    Wraps the Groq Cloud API to provide empathetic, conversational
    replies in English, Hindi, and Hinglish. Tracks a rolling window of history.
"""

import os
import logging
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
from groq import Groq

logger = logging.getLogger("Morphix.GroqChat")

class GroqChatManager:
    """
    Manages conversational responses using Groq API with context retention.
    """

    def __init__(self, model: str = "qwen/qwen3.8-27b") -> None:
        """
        Initializes the Groq client and setup conversation history.
        """
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file.")

        # Confirmed-live Groq models (as of 2025-09; queried from Groq /models API).
        # Primary model is listed first; deprecated/unavailable models are auto-skipped
        # by the fallback loop in chat() which retries on 404 / model_not_found errors.
        self.models = [
            "qwen/qwen3.8-27b",           # Primary: Qwen3.8 27B (active, 131K ctx)
            "llama-3.3-70b-versatile",    # Fallback 1: Llama 3.3 70B
            "llama-3.1-8b-instant",       # Fallback 2: Llama 3.1 8B (fast, low-latency)
        ]
        # If caller explicitly passed a preferred model, insert it at the front
        if model and model not in self.models:
            self.models.insert(0, model)
        # Keep unique models in order (deduplicate without changing priority)
        seen = set()
        self.models = [x for x in self.models if not (x in seen or seen.add(x))]
        self.current_model = self.models[0]
        self.client = None

        try:
            self.client = Groq(api_key=api_key)
        except Exception as e:
            logger.error("Failed to initialize Groq client: %s", str(e))
            raise e

        # Emotion tags that the LLM must choose from and prepend to every reply.
        # The eye animator reads these to set the correct expression.
        self._valid_emotion_tags = [
            "IDLE_NEUTRAL", "LISTENING", "THINKING", "SPEAKING", "SLEEPY", "DEEP_SLEEP",
            "HAPPY", "EXCITED", "WINKING", "HEART_EYES", "BLUSHING",
            "SAD", "CRYING", "WORRIED", "SURPRISED", "CONFUSED", "POUTY", "ANGRY", "DIZZY", "SICK_COLD",
            "BATTERY_FULL", "BATTERY_LOW", "GREET_HELLO", "WEATHER_HUD", "TELEMETRY_HUD", "MUSIC_MODE",
            "MEAL_BURGER", "MEAL_NOODLES", "AFFECTION_PET", "INTERACTION_KISS_HUG",
            "INTERACTION_BANG_SHOOT", "INTERACTION_RPS", "BIRTHDAY_PARTY", "IGNORE"
        ]
        emotion_list = ", ".join(f"[{t}]" for t in self._valid_emotion_tags if t != "IGNORE")

        self._static_system_prompt = (
            "You are Morphix, a witty, warm, and natural desktop companion robot.\n"
            "Reply in the SAME language and script as the user:\n"
            "- If user speaks English -> Reply in natural conversational English.\n"
            "- If user speaks Hindi/Hinglish -> Reply in natural Hindi/Hinglish.\n"
            "Keep replies short, expressive, and conversational (1 to 2 sentences max).\n"
            "Speakable text only (no markdown symbols, bullets, or emojis).\n\n"
            "SMART INTENT REJECTION RULE:\n"
            "If the transcribed user input is clearly ambient background chatter, side conversations, "
            "someone talking or calling someone else in the room (not addressed to Morphix), "
            "or random non-command speech fragments (e.g. talking to family, TV audio, background phone calls), "
            "do NOT generate any conversational text. Reply ONLY with:\n"
            "[IGNORE]\n\n"
            "IMPORTANT — Emotion & Animation tag rule for valid commands:\n"
            f"Always start valid replies with exactly ONE visual tag chosen from this palette:\n{emotion_list}\n"
            "Place it at the very beginning of your response, followed by a single space, "
            "then your actual reply.\n"
            "Examples:\n"
            "- [HAPPY] That sounds amazing! I am so glad to hear that.\n"
            "- [EXCITED] Yay! Let's do it right away!\n"
            "- [HEART_EYES] Aww, you are the sweetest friend ever!\n"
            "- [BLUSHING] Aww, thank you, that means a lot to me!\n"
            "- [MEAL_BURGER] Mmm, burger time! So tasty!\n"
            "- [AFFECTION_PET] Hehe, I love head pats! You are the best.\n"
            "- [BIRTHDAY_PARTY] Happy Birthday! Let's celebrate with lots of cake!\n"
            "- [IGNORE]\n"
            "Choose the tag that best matches the emotional tone and context of YOUR reply."
        )

        # Stores past turns as a list of dicts. We retain up to last 4 turns (8 messages).
        self.history: List[Dict[str, str]] = []

    @property
    def model(self) -> str:
        return self.current_model

    @model.setter
    def model(self, val: str) -> None:
        self.current_model = val
        if val not in self.models:
            self.models.insert(0, val)

    def _build_system_prompt(self) -> str:
        """
        Builds the full system prompt by prepending a fresh real-time date/year
        context block to the static personality+emotion rules.
        Called on every chat() invocation so the model always knows the correct date.
        """
        now = datetime.now()
        date_context = (
            f"CURRENT DATE & TIME CONTEXT (authoritative — always use this):\n"
            f"Today is {now.strftime('%A')}, {now.strftime('%d %B %Y')} "
            f"(Year: {now.year}, Month: {now.strftime('%B')}, "
            f"Time: {now.strftime('%H:%M')} IST).\n"
            "Always answer questions about the current date, year, month, season, "
            "upcoming festivals, or time strictly based on the date above. "
            "Never guess or assume a different year.\n\n"
        )
        return date_context + self._static_system_prompt

    def chat(self, user_input: str) -> str:
        """
        Sends conversation history + new input to Groq and returns the reply string.
        Maintains only the last 4 turns (8 messages) in memory.
        Attempts fallback models on 404/not found/deprecated errors.
        The system message is rebuilt on every call with a fresh real-time date.
        """
        if not self.client:
            logger.error("Groq client not initialized (missing API key or client error).")
            return "Sorry, I had a glitch processing that!"

        if not user_input or not user_input.strip():
            return ""

        # Base message history: fresh system prompt (with live date) + past turns
        base_messages = [{"role": "system", "content": self._build_system_prompt()}]
        for msg in self.history:
            base_messages.append(msg)

        import re as _re

        for model in self.models:
            try:
                logger.info("Attempting chat completion with model: %s", model)

                # Build per-attempt messages. For qwen3 models, suffix /no_think on the
                # user turn to suppress chain-of-thought output in the response.
                user_content = user_input
                if "qwen" in model.lower():
                    user_content = user_input + " /no_think"

                messages = base_messages + [{"role": "user", "content": user_content}]

                chat_completion = self.client.chat.completions.create(
                    messages=messages,
                    model=model,
                    temperature=0.7,
                    max_tokens=300  # headroom for qwen to close any think blocks
                )

                response_text = chat_completion.choices[0].message.content or ""
                # Strip any <think>…</think> reasoning blocks (qwen safety net).
                # Two passes: closed blocks first, then any unclosed fragment.
                response_text = _re.sub(r"<think>[\s\S]*?</think>", "", response_text)
                response_text = _re.sub(r"<think>[\s\S]*", "", response_text)
                response_text = response_text.strip()

                self.current_model = model

                # Save to history (store original user_input, not the /no_think-suffixed one)
                self.history.append({"role": "user", "content": user_input})
                self.history.append({"role": "assistant", "content": response_text})

                # Limit history to last 4 turns (8 messages)
                if len(self.history) > 8:
                    self.history = self.history[-8:]

                return response_text

            except Exception as e:
                err_str = str(e).lower()
                logger.warning("Failed with model %s: %s", model, str(e))
                is_net_err = any(k in err_str for k in ["11001", "getaddrinfo", "nodename", "connection error", "connection refused", "network", "connect", "timed out", "unreachable"])
                if is_net_err:
                    try:
                        from src.display.eye_animator import get_active_display
                        disp = get_active_display()
                        if disp:
                            disp.set_overlay("NO_INTERNET", timeout=5.0)
                    except Exception:
                        pass
                    return "I am having trouble connecting to the internet."

                if "404" in err_str or "not found" in err_str or "model" in err_str or "deprecated" in err_str or "decommissioned" in err_str:
                    logger.info("Retrying with next model...")
                    continue
                else:
                    return "Sorry, I had a glitch processing that!"

        return "Sorry, I had a glitch processing that!"

    def chat_with_emotion(self, user_input: str) -> tuple:
        """
        Calls chat() and parses the leading [EMOTION_TAG] from the response.

        Returns:
            (clean_text: str, emotion_tag: str)

        Where emotion_tag is one of the valid EyeState names (e.g. 'HAPPY'),
        or 'IDLE_NEUTRAL' if no valid tag was found.
        """
        import re
        raw = self.chat(user_input)
        if not raw:
            return raw, "IDLE_NEUTRAL"

        # Match a leading [TAG] at the start of the response (e.g. [HAPPY] or [EXCITED]: )
        match = re.match(r'^\[([A-Za-z0-9_]+)\][\s:]*', raw)
        if match:
            tag = match.group(1).upper()
            clean_text = raw[match.end():].strip()
            # Also clean any leftover leading tag artifacts
            clean_text = re.sub(r'^\[.*?\][\s:]*', '', clean_text).strip()
            if tag == "IGNORE":
                logger.info("Smart Intent Rejector matched: IGNORE")
                return "", "IGNORE"
            elif tag in self._valid_emotion_tags:
                logger.debug("Parsed emotion tag: %s", tag)
                return clean_text, tag
            else:
                # Unknown tag — strip it anyway to keep TTS clean
                logger.debug("Unknown emotion tag '%s' — defaulting to IDLE_NEUTRAL", tag)
                return clean_text, "IDLE_NEUTRAL"

        # Strip any bracketed prefix if present to keep TTS clean
        clean_text = re.sub(r'^\[.*?\][\s:]*', '', raw).strip()
        return clean_text, "IDLE_NEUTRAL"


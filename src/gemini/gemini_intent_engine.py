"""
Morphix Desktop Companion Robot - Gemini Intent Engine
======================================================
File: src/gemini/gemini_intent_engine.py
Description:
    Manages structured intent extraction, chat completions, retry attempts,
    timeout constraints, and request deduplications for Whisper outputs.
"""

import logging
import random
import re
import time
from typing import Any, Dict, Optional

from google.genai import types
from src.gemini.chat_memory import ChatMemory
from src.gemini.gemini_client import GeminiClient
from src.gemini.gemini_config import GeminiConfig
from src.gemini.gemini_logger import log_gemini_event
from src.gemini.json_parser import JsonParser
from src.gemini.prompt_builder import PromptBuilder
from src.gemini.response_validator import ResponseValidator

logger = logging.getLogger("Morphix.GeminiEngine")


class GeminiIntentEngine:
    """
    Coordinates API generation requests for intent routing and companion chat.
    """

    def __init__(self, client: GeminiClient, memory: Optional[ChatMemory] = None) -> None:
        self.client = client
        self.memory = memory or ChatMemory()

        # Request deduplication variables
        self.last_transcript: Optional[str] = None
        self.last_transcript_time: float = 0.0
        self.last_result: Dict[str, Any] = {"intent": "UNKNOWN"}

        # Rate limit circuit breaker state
        self._consecutive_429s: int = 0
        self._rate_limit_cooldown_until: float = 0.0

    def _is_rate_limited(self) -> bool:
        """Returns True if we are in the circuit-breaker cooldown window."""
        if self._consecutive_429s >= GeminiConfig.RATE_LIMIT_CONSECUTIVE_THRESHOLD:
            remaining = self._rate_limit_cooldown_until - time.time()
            if remaining > 0:
                logger.warning(
                    "[RateLimit] Circuit breaker active. Skipping Gemini for %.0fs more. Falling back to local.",
                    remaining
                )
                return True
            else:
                # Cooldown expired — reset and allow retrying
                logger.info("[RateLimit] Cooldown expired. Resuming Gemini requests.")
                self._consecutive_429s = 0
        return False

    @staticmethod
    def _is_429_error(error: Exception) -> bool:
        """Detects a 429 Too Many Requests / RESOURCE_EXHAUSTED error."""
        error_str = str(error)
        return (
            "429" in error_str
            or "RESOURCE_EXHAUSTED" in error_str
            or "Too Many Requests" in error_str
            or "quota" in error_str.lower()
        )

    @staticmethod
    def _extract_retry_after(error: Exception) -> Optional[float]:
        """Tries to parse a Retry-After delay (in seconds) from the error message."""
        error_str = str(error)
        # Look for 'retryDelay: "Xs"' pattern in Google API error bodies
        match = re.search(r'retryDelay["\s:]+([\d.]+)', error_str)
        if match:
            return float(match.group(1))
        # Also try plain 'retry_delay: X' or 'Retry-After: X'
        match = re.search(r'[Rr]etry[_-][Aa]fter["\s:]+([\d.]+)', error_str)
        if match:
            return float(match.group(1))
        return None

    def process_transcript(self, transcript: str) -> Dict[str, Any]:
        """
        Extracts structured intent with retry, exponential backoff on 429 rate limits,
        circuit-breaker cooldown, timeout constraints, and deduplication windows.

        :param transcript: Raw Cleaned transcript.
        :return: Validated intent dictionary.
        """
        if not transcript:
            return {"intent": "UNKNOWN"}

        # Request Deduplication Check
        now = time.time()
        if self.last_transcript == transcript and (now - self.last_transcript_time) < GeminiConfig.DEDUPLICATION_WINDOW_SECONDS:
            logger.info("Ignoring duplicate API request for transcript: '%s'", transcript)
            # Log as LOCAL/CACHED since we bypass calling Gemini API
            log_gemini_event(
                request_type="intent",
                intent=self.last_result.get("intent", "UNKNOWN"),
                latency=0.0,
                success=True,
                model_name=self.client.model_name,
                source="LOCAL"
            )
            return self.last_result

        # Update cache trackers
        self.last_transcript = transcript
        self.last_transcript_time = now

        if not self.client.is_available() or not self.client.validate_key_if_first_time():
            logger.warning("Gemini Client not available. Disabling engine processing.")
            self.last_result = {"intent": "UNKNOWN"}
            return self.last_result

        # Circuit breaker: skip Gemini entirely during cooldown window
        if self._is_rate_limited():
            self.last_result = {"intent": "UNKNOWN"}
            return self.last_result

        system_instruction = PromptBuilder.get_intent_system_prompt()
        prompt = f"Transcript: {transcript}"
        max_attempts = GeminiConfig.MAX_RETRY_COUNT + 1

        for attempt in range(max_attempts):
            start_time = time.time()
            try:
                config = types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=GeminiConfig.INTENT_TEMPERATURE,
                    max_output_tokens=GeminiConfig.MAX_OUTPUT_TOKENS,
                    response_mime_type="application/json"
                )

                api_start = time.time()
                response = self.client.client.models.generate_content(
                    model=self.client.model_name,
                    contents=prompt,
                    config=config
                )
                inference_time = time.time() - api_start
                latency = time.time() - start_time
                raw_text = response.text
                parsed = JsonParser.parse_gemini_json(raw_text)

                if parsed and ResponseValidator.validate_intent_json(parsed):
                    # Advisory check for confidence threshold
                    confidence = parsed.get("confidence", 0.0)
                    if confidence < GeminiConfig.CONFIDENCE_THRESHOLD:
                        logger.info(
                            "Gemini confidence %.2f is below threshold %.2f. Mapping intent to UNKNOWN.",
                            confidence, GeminiConfig.CONFIDENCE_THRESHOLD
                        )
                        parsed["intent"] = "UNKNOWN"

                    # Successful call — reset rate limit counter
                    self._consecutive_429s = 0

                    log_gemini_event(
                        request_type="intent",
                        intent=parsed.get("intent", "UNKNOWN"),
                        latency=latency,
                        success=True,
                        model_name=self.client.model_name,
                        retry_count=attempt,
                        validation_result="PASS",
                        response_length=len(raw_text) if raw_text else 0,
                        input_length=len(prompt),
                        source="GEMINI",
                        inference_time=inference_time
                    )
                    self.last_result = parsed
                    return self.last_result
                else:
                    log_gemini_event(
                        request_type="intent",
                        intent="UNKNOWN",
                        latency=latency,
                        success=False,
                        model_name=self.client.model_name,
                        retry_count=attempt,
                        validation_result="FAIL",
                        response_length=len(raw_text) if raw_text else 0,
                        input_length=len(prompt),
                        error_msg="Validation schema check failed.",
                        source="GEMINI",
                        inference_time=inference_time
                    )

            except Exception as e:
                latency = time.time() - start_time

                if self._is_429_error(e):
                    self._consecutive_429s += 1

                    # Engage circuit breaker if threshold exceeded
                    if self._consecutive_429s >= GeminiConfig.RATE_LIMIT_CONSECUTIVE_THRESHOLD:
                        self._rate_limit_cooldown_until = time.time() + GeminiConfig.RATE_LIMIT_COOLDOWN_SECONDS
                        logger.warning(
                            "[RateLimit] %d consecutive 429s — circuit breaker engaged. "
                            "Pausing Gemini for %.0fs. Falling back to local parsers.",
                            self._consecutive_429s, GeminiConfig.RATE_LIMIT_COOLDOWN_SECONDS
                        )
                        log_gemini_event(
                            request_type="intent",
                            intent="UNKNOWN",
                            latency=latency,
                            success=False,
                            model_name=self.client.model_name,
                            retry_count=attempt,
                            validation_result="FAIL",
                            error_msg="429 RESOURCE_EXHAUSTED — circuit breaker engaged",
                            source="GEMINI",
                            inference_time=latency
                        )
                        break  # Exit retry loop immediately, fall back to local

                    # Exponential backoff with jitter before next retry
                    # Try to honor Retry-After from the error if present
                    retry_after = self._extract_retry_after(e)
                    if retry_after:
                        backoff = min(retry_after, GeminiConfig.RATE_LIMIT_BACKOFF_MAX)
                        logger.warning(
                            "[RateLimit] 429 received (attempt %d/%d). Honoring Retry-After: %.1fs.",
                            attempt + 1, max_attempts, backoff
                        )
                    else:
                        # Exponential backoff: base * 2^attempt + random jitter
                        backoff = min(
                            GeminiConfig.RATE_LIMIT_BACKOFF_BASE * (2 ** attempt)
                            + random.uniform(0.0, 1.0),
                            GeminiConfig.RATE_LIMIT_BACKOFF_MAX
                        )
                        logger.warning(
                            "[RateLimit] 429 received (attempt %d/%d). Backing off %.1fs before retry.",
                            attempt + 1, max_attempts, backoff
                        )

                    log_gemini_event(
                        request_type="intent",
                        intent="UNKNOWN",
                        latency=latency,
                        success=False,
                        model_name=self.client.model_name,
                        retry_count=attempt,
                        validation_result="FAIL",
                        error_msg=f"429 RESOURCE_EXHAUSTED — retrying in {backoff:.1f}s",
                        source="GEMINI",
                        inference_time=latency
                    )

                    if attempt < max_attempts - 1:
                        time.sleep(backoff)

                else:
                    # Non-rate-limit error — log and continue to next attempt
                    log_gemini_event(
                        request_type="intent",
                        intent="UNKNOWN",
                        latency=latency,
                        success=False,
                        model_name=self.client.model_name,
                        retry_count=attempt,
                        validation_result="FAIL",
                        error_msg=str(e),
                        source="GEMINI",
                        inference_time=latency
                    )

        self.last_result = {"intent": "UNKNOWN"}
        return self.last_result

    def generate_chat_response(self, user_query: str) -> str:
        """
        Generates general conversational response using rolling memory context.

        :param user_query: User raw input query.
        :return: Spoken companion robotic reply.
        """
        if not self.client.is_available() or not self.client.validate_key_if_first_time():
            return "I'm sorry, I couldn't process that request right now."

        start_time = time.time()
        system_instruction = PromptBuilder.get_chat_system_prompt()

        try:
            # Build rolling memory content formats
            contents = []
            for msg in self.memory.get_history():
                contents.append(
                    types.Content(
                        role=msg["role"],
                        parts=[types.Part.from_text(text=msg["parts"][0])]
                    )
                )
            # Append current turn query
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=user_query)]
                )
            )

            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=GeminiConfig.CHAT_TEMPERATURE,
                max_output_tokens=GeminiConfig.MAX_OUTPUT_TOKENS
            )

            api_start = time.time()
            response = self.client.client.models.generate_content(
                model=self.client.model_name,
                contents=contents,
                config=config
            )
            inference_time = time.time() - api_start
            response_text = response.text.strip()
            latency = time.time() - start_time

            log_gemini_event(
                request_type="chat",
                intent="CHAT",
                latency=latency,
                success=True,
                model_name=self.client.model_name,
                response_length=len(response_text),
                input_length=len(user_query),
                source="GEMINI",
                inference_time=inference_time
            )

            # Store in conversation memory history
            self.memory.add_exchange(user_query, response_text)
            return response_text

        except Exception as e:
            latency = time.time() - start_time
            if self._is_429_error(e):
                self._consecutive_429s += 1
                log_gemini_event(
                    request_type="chat",
                    intent="CHAT",
                    latency=latency,
                    success=False,
                    model_name=self.client.model_name,
                    error_msg="429 RESOURCE_EXHAUSTED",
                    source="GEMINI",
                    inference_time=latency
                )
                return "I'm a bit overwhelmed right now — too many requests. Please try again in a moment."
            else:
                log_gemini_event(
                    request_type="chat",
                    intent="CHAT",
                    latency=latency,
                    success=False,
                    model_name=self.client.model_name,
                    error_msg=str(e),
                    source="GEMINI",
                    inference_time=latency
                )
                return "I'm sorry, I couldn't process that request right now."

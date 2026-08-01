"""
Morphix Desktop Companion Robot - Gemini Intent Engine
======================================================
File: src/gemini/gemini_intent_engine.py
Description:
    Manages structured intent extraction, chat completions, retry attempts,
    timeout constraints, and request deduplications for Whisper outputs.
"""

import logging
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

    def process_transcript(self, transcript: str) -> Dict[str, Any]:
        """
        Extracts structured intent with retry, timeout limits, and deduplication windows.

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
                        logger.info("Gemini confidence %.2f is below threshold %.2f. Mapping intent to UNKNOWN.", confidence, GeminiConfig.CONFIDENCE_THRESHOLD)
                        parsed["intent"] = "UNKNOWN"

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

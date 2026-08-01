"""
Morphix Desktop Companion Robot - Gemini Client
===============================================
File: src/gemini/gemini_client.py
Description:
    Establishes connection to the Google Gemini model using the official modern
    Google Gen AI SDK. Safely disables itself if credentials are not present.
"""

import logging
import os
from typing import Optional, Tuple

from google import genai
from src.gemini.gemini_config import GeminiConfig

logger = logging.getLogger("Morphix.GeminiClient")


def validate_api_key(api_key: Optional[str]) -> Tuple[bool, str]:
    """
    Validates the structure of the Gemini API key.
    Returns (is_valid, reason_if_invalid).
    """
    if not api_key:
        return False, "GEMINI_API_KEY environment variable is missing or empty"
        
    cleaned = api_key.strip()
    if not cleaned:
        return False, "GEMINI_API_KEY environment variable is empty or whitespace"
        
    placeholders = {
        "your_api_key", "your_gemini_api_key", "insert_key_here", "your_api_key_here",
        "placeholder", "your-api-key", "xxxx"
    }
    if cleaned.lower() in placeholders or "placeholder" in cleaned.lower():
        return False, "GEMINI_API_KEY contains a placeholder value"
        
    if len(cleaned) < 15:
        return False, "GEMINI_API_KEY is too short (must be at least 15 characters)"
        
    # Check expected prefixes: Gemini API keys normally start with 'AIzaSy'
    # We also support 'test_' prefix for local mock testing
    if not (cleaned.startswith("AIzaSy") or cleaned.startswith("test_")):
        return False, "GEMINI_API_KEY does not start with expected prefix ('AIzaSy' or 'test_')"
        
    return True, ""


class GeminiClient:
    """
    Initializes modern GenAI Client safely without hardcoded keys.
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = model_name or GeminiConfig.DEFAULT_MODEL_NAME
        self.client = None
        self._checked_validation = False
        self._disable_reason = ""

        is_valid, reason = validate_api_key(self.api_key)

        if is_valid:
            try:
                # Initialize using the modern SDK with configured timeout HTTP options
                from google.genai import types
                timeout_ms = int(GeminiConfig.TIMEOUT_SECONDS * 1000)
                self.client = genai.Client(
                    api_key=self.api_key,
                    http_options=types.HttpOptions(timeout=timeout_ms)
                )
                logger.info("GenAI Client created successfully. Model: '%s'", self.model_name)
            except Exception as e:
                self._disable_reason = f"Initialization error: {str(e)}"
                logger.warning("Failed to initialize modern GenAI client: %s", str(e))
        else:
            self._disable_reason = reason
            print(f"\nWARNING: Gemini NLU features will be disabled. Reason: {reason}\n")
            logger.warning("Gemini Client disabled. Reason: %s", reason)

    def is_available(self) -> bool:
        """Returns True if client successfully initialized."""
        return self.client is not None

    def validate_key_if_first_time(self) -> bool:
        """
        Validates the key on the first real API call.
        If validation fails, disables the client and returns False.
        """
        if self.client is None:
            return False

        if getattr(self, "_checked_validation", False):
            return self.client is not None

        self._checked_validation = True

        try:
            # Test credentials using a lightweight list models request
            next(iter(self.client.models.list(config={'page_size': 1})), None)
            return True
        except Exception as e:
            error_str = str(e)
            # Categorize the error for clear, actionable output
            if "API_KEY_INVALID" in error_str or "API key not valid" in error_str:
                reason = "API key is invalid or has been revoked"
                hint = "Get a new key at: https://aistudio.google.com/app/apikey"
            elif "INVALID_ARGUMENT" in error_str:
                reason = "API key format is rejected by Google (ensure it starts with 'AIzaSy')"
                hint = "Get a new key at: https://aistudio.google.com/app/apikey"
            elif "PERMISSION_DENIED" in error_str:
                reason = "API key does not have Gemini API access"
                hint = "Enable the Generative Language API in your Google Cloud project"
            else:
                reason = "Connection or authentication failure"
                hint = "Check your internet connection and API key"

            print(f"\n[Gemini]: Disabling AI features — {reason}.")
            print(f"[Gemini]: {hint}\n")
            logger.error("Gemini API Key validation failed. Disabling Gemini. Reason: %s", reason)
            self.client = None
            return False

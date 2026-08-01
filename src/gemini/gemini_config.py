"""
Morphix Desktop Companion Robot - Gemini Configuration
======================================================
File: src/gemini/gemini_config.py
Description:
    Centralizes environment settings, default model selections, and
    hyperparameters for the Gemini AI Brain.
"""

import os


class GeminiConfig:
    """
    Constant settings and validation configuration constraints for Gemini.
    """
    # Prevent collision with Whisper configuration (e.g. MODEL=tiny.en)
    DEFAULT_MODEL_NAME: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    INTENT_TEMPERATURE: float = 0.0
    CHAT_TEMPERATURE: float = 0.7
    TIMEOUT_SECONDS: float = 8.0
    MAX_RETRY_COUNT: int = 2

    # Rate limit (429) backoff settings
    # Base delay in seconds before first retry after a 429
    RATE_LIMIT_BACKOFF_BASE: float = 2.0
    # Maximum delay cap (seconds) so we never wait too long in a voice assistant
    RATE_LIMIT_BACKOFF_MAX: float = 10.0
    # After this many consecutive 429s, stop trying Gemini for RATE_LIMIT_COOLDOWN_SECONDS
    RATE_LIMIT_CONSECUTIVE_THRESHOLD: int = 3
    RATE_LIMIT_COOLDOWN_SECONDS: float = 60.0

    # Max memory limit: 5 user messages and 5 assistant messages
    CHAT_HISTORY_LIMIT: int = 10

    MAX_OUTPUT_TOKENS: int = 512
    CONFIDENCE_THRESHOLD: float = 0.7
    DEDUPLICATION_WINDOW_SECONDS: float = 3.0
    COMMAND_QUEUE_WARMUP_TIMEOUT: float = 1.0

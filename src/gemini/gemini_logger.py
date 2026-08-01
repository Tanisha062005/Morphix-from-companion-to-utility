"""
Morphix Desktop Companion Robot - Gemini Logger
================================================
File: src/gemini/gemini_logger.py
Description:
    Records metrics including model, latency, inputs/outputs, retries,
    validation successes, and intent sources (LOCAL vs GEMINI).
"""

from datetime import datetime
import logging
import sys
from typing import Optional

# Setup logger for Gemini interactions
logger = logging.getLogger("Morphix.GeminiLog")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [Gemini]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def log_gemini_event(
    request_type: str,
    intent: str,
    latency: float,
    success: bool,
    model_name: str,
    retry_count: int = 0,
    validation_result: str = "PASS",
    response_length: int = 0,
    input_length: int = 0,
    error_msg: Optional[str] = None,
    source: str = "GEMINI",
    inference_time: Optional[float] = None
) -> None:
    """
    Safely logs API metrics without printing keys or user data.
    """
    timestamp = datetime.now().isoformat()
    inf_time = inference_time if inference_time is not None else latency
    log_str = (
        f"Timestamp: {timestamp} | Model: {model_name} | Type: {request_type} | "
        f"Intent: {intent} | Source: {source} | Latency: {latency:.4f}s | "
        f"InferenceTime: {inf_time:.4f}s | Success: {success} | Retries: {retry_count} | "
        f"Validation: {validation_result} | InLen: {input_length} | OutLen: {response_length}"
    )
    if error_msg:
        log_str += f" | Error: {error_msg}"

    if success:
        logger.info(log_str)
    else:
        logger.error(log_str)

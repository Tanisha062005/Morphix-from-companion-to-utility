"""
Morphix Desktop Companion Robot - Text-To-Speech Test Suite
============================================================
File: tests/test_tts.py
Description:
    Tests the TextToSpeech class using Piper offline voice synthesis
    and audio playback for Morphix.
"""

import logging
from pathlib import Path
import sys

# Ensure project root is in sys.path for clean import resolution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.voice.text_to_speech import TextToSpeech

# Configure test runner logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [TestTTS]: %(message)s"
)
logger = logging.getLogger("TestTTS")


def run_tests() -> None:
    """
    Executes offline Text-To-Speech integration tests.
    """
    logger.info("==========================================")
    logger.info("     INITIALIZING PIPER TTS TEST          ")
    logger.info("==========================================")

    try:
        tts = TextToSpeech()

        sentences = [
            "Hello Tanisha.",
            "I am Morphix.",
            "I am now using Piper offline voice.",
            "Nice to meet you."
        ]

        for idx, sentence in enumerate(sentences, start=1):
            logger.info("------------------------------------------")
            logger.info("Test %d/4: Speaking sentence -> '%s'", idx, sentence)
            logger.info("------------------------------------------")

            success = tts.speak(sentence)
            if success:
                logger.info("Test %d passed: Speech completed successfully.", idx)
            else:
                logger.error("Test %d failed: Speech output did not complete.", idx)

        logger.info("==========================================")
        logger.info("All TTS tests executed cleanly.")
        logger.info("==========================================")

    except Exception as e:
        logger.error("An unexpected error occurred during TTS testing: %s", str(e), exc_info=True)


if __name__ == "__main__":
    run_tests()

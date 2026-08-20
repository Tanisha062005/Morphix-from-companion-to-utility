"""
REMOVED — WhisperEngine has been removed from Morphix.

Morphix now uses Google Speech Recognition exclusively for STT.
See src/voice/speech_to_text.py → GoogleSpeechEngine.
"""

raise ImportError(
    "WhisperEngine has been removed. "
    "Use src.voice.speech_to_text.SpeechToText (backed by GoogleSpeechEngine) instead."
)

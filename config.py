
"""
Morphix Desktop Companion Robot - Global Configuration
======================================================
File: config.py
Description:
    Centralized configuration file storing global constants and system settings 
    for the Morphix desktop companion robot project.
"""


import os


class Config:
    """
    Stores global project configuration settings and default values for Morphix.
    """

    # -------------------------------------------------------------------------
    # 1. Robot Information
    # -------------------------------------------------------------------------
    ROBOT_NAME: str = "Morphix"
    ROBOT_VERSION: str = "1.0.0"
    DEVELOPER_NAME: str = "Developer"
    GEMINI_ENABLED: bool = False

    # -------------------------------------------------------------------------
    # 2. Personality Settings
    # -------------------------------------------------------------------------
    DEFAULT_MOOD: str = "Happy"
    DEFAULT_LANGUAGE: str = "English"
    DEFAULT_SPEAKING_STYLE: str = "Friendly, curious, and playful"

    # -------------------------------------------------------------------------
    # 3. Voice Settings
    # -------------------------------------------------------------------------
    VOICE_ENABLED: bool = True
    OFFLINE_MODE: bool = True
    WAKE_RESPONSE: str = "Listening."
    WAKE_SOUND: str = "assets/sounds/morphix_voice.wav"
    STT_MODEL: str = os.getenv("MODEL", "base.en")
    WHISPER_MODELS_DIR: str = "models/whisper"
    WAKE_WORD_THRESHOLD: float = 82.0
    ACTIVE_MODE_TIMEOUT: float = 20.0

    # Wake Word Pipeline Configuration
    WAKE_PHRASE: str = "hey morphix"
    SIMILARITY_THRESHOLD: float = 0.62
    WAKE_PHRASES: list = [
        "hey morphix",
        "hello morphix",
        "hi morphix",
        "morphix",
        "wake up morphix",
        "hey morfix",
        "hello morfix",
        "morphics",
        "hey morphics",
        "he might fix",
        "hey more fix",
        "hey more pics",
        "more fix"
    ]
    ACCEPTED_ALIASES: list = [
        "hey morphix",
        "hey morfix",
        "hey morphics",
        "hey more fix",
        "hey more pics",
        "he might fix",
        "he morfix",
        "he morphics",
        "he more fix",
        "he more",
        "he motivates",
        "he wont fix",
        "hey morefiks",
        "morphix",
    ]




    # -------------------------------------------------------------------------
    # 4. Display Settings
    # -------------------------------------------------------------------------
    DISPLAY_ENABLED: bool = True
    DEFAULT_BRIGHTNESS: int = 80  # Scale 0 to 100

    # -------------------------------------------------------------------------
    # 5. Camera Settings
    # -------------------------------------------------------------------------
    CAMERA_ENABLED: bool = True

    # -------------------------------------------------------------------------
    # 6. Reminder Settings
    # -------------------------------------------------------------------------
    REMINDER_ENABLED: bool = True

    # -------------------------------------------------------------------------
    # 7. Surveillance Settings
    # -------------------------------------------------------------------------
    SURVEILLANCE_ENABLED: bool = False

    # -------------------------------------------------------------------------
    # 8. Debug Settings
    # -------------------------------------------------------------------------
    DEBUG_MODE: bool = True
    LOGGING_ENABLED: bool = True
    DEVELOPMENT_MODE: bool = False
    DEBUG_AUDIO_PIPELINE: bool = False
    COMMAND_QUEUE_WARMUP_TIMEOUT: float = 1.0



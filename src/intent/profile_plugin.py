"""
Morphix Desktop Companion Robot - Voice Profile Intent Plugin
=============================================================
File: src/intent/profile_plugin.py
Description:
    Provides natural voice onboarding for user identity configuration:
    1. Sets and queries user name ("My name is Alex", "Mera naam Alex hai", "Who am I?").
    2. Triggers voice face enrollment ("Register my face", "Mera face scan karo").
"""

import logging
import re
from typing import Any, Dict, Optional, Tuple

from src.config.profile_manager import get_primary_user, set_primary_user
from src.intent.intent_classifier import IntentResult, IntentType
from src.vision.camera_manager import CameraManager

logger = logging.getLogger("Morphix.ProfilePlugin")


class ProfileService:
    """
    Executes profile management business logic and formats voice responses.
    """

    _instance: Optional["ProfileService"] = None

    @classmethod
    def get_instance(cls) -> "ProfileService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_user_name(self, name: str, is_hindi: bool = False) -> str:
        """
        Updates primary user name in config and returns spoken confirmation.

        :param name: New user name.
        :param is_hindi: True for Hindi response phrasing.
        :return: Spoken response string.
        """
        cleaned_name = name.strip().title()
        success = set_primary_user(cleaned_name)
        if success:
            if is_hindi:
                return f"Theek hai! Ab se main aapko {cleaned_name} bulaungi."
            return f"Got it! I will call you {cleaned_name} from now on."
        else:
            if is_hindi:
                return "Kshama karein, main naam update nahi kar payi."
            return "Sorry, I could not update your name."

    def get_user_name(self, is_hindi: bool = False) -> str:
        """
        Retrieves current primary user name and returns spoken answer.

        :param is_hindi: True for Hindi response phrasing.
        :return: Spoken response string.
        """
        current_name = get_primary_user()
        if is_hindi:
            return f"Aapka naam {current_name} hai."
        return f"You are {current_name}."

    def enroll_user_face(self, is_hindi: bool = False) -> str:
        """
        Captures face photos for current primary user and reloads face dataset model.

        :param is_hindi: True for Hindi response phrasing.
        :return: Spoken response string.
        """
        user_name = get_primary_user()
        cam = CameraManager.get_instance()
        success, saved_files, msg = cam.capture_face_samples(user_name, count=6)

        if success and len(saved_files) > 0:
            if is_hindi:
                return f"Aapka face successfully register ho gaya hai, {user_name}!"
            return f"Your face has been registered successfully, {user_name}!"
        else:
            if is_hindi:
                return f"Face capture nahi ho paya. Kripya camera check karein."
            return f"Could not register your face. Please ensure camera is connected."


class ProfileIntentPlugin:
    """
    Regex-based Intent Plugin handling voice profile setup and face enrollment.
    """

    def __init__(self) -> None:
        self.service = ProfileService.get_instance()

    def match(self, text: str) -> Optional[IntentResult]:
        """
        Matches user commands against voice profile and face registration patterns.

        :param text: Raw spoken input string.
        :return: IntentResult if matched, None otherwise.
        """
        if not text or not isinstance(text, str):
            return None

        cleaned = text.strip()
        lowered = cleaned.lower()

        # 1. FACE ENROLLMENT INTENT
        enroll_patterns_hi = [
            r"^(?:mera\s+)?(?:face|chehra)\s+(?:scan|register|enroll|save|capture)(?:\s+(?:kar\s+do|karo|kijiye))?$",
            r"^(?:scan|register|enroll|save|capture)\s+(?:mera\s+)?(?:face|chehra)$",
        ]
        for pat in enroll_patterns_hi:
            if re.search(pat, lowered):
                return IntentResult(
                    intent=IntentType.ENROLL_FACE,
                    parameters={"is_hindi": True},
                    confidence=0.98,
                    raw_text=cleaned,
                )

        enroll_patterns_en = [
            r"^(?:register|enroll|capture|scan|save)\s+(?:my\s+)?face$",
            r"^take\s+(?:my\s+)?face\s+(?:photo|photos|picture|sample|samples)$",
            r"^add\s+(?:my\s+)?face(?:\s+to\s+known\s+faces)?$",
        ]
        for pat in enroll_patterns_en:
            if re.search(pat, lowered):
                return IntentResult(
                    intent=IntentType.ENROLL_FACE,
                    parameters={"is_hindi": False},
                    confidence=0.98,
                    raw_text=cleaned,
                )

        # 2. SET USER NAME INTENT
        # Hindi patterns
        set_name_patterns_hi = [
            r"^(?:mera\s+naam|naam)\s+(?P<name>[a-zA-Z\u0900-\u097F]+)(?:\s+hai|\s+rakho|\s+set\s+karo)?$",
            r"^mujhe\s+(?P<name>[a-zA-Z\u0900-\u097F]+)\s+(?:bulao|kaho)$",
            r"^mera\s+naam\s+badal\s+kar\s+(?P<name>[a-zA-Z\u0900-\u097F]+)\s+karo$",
        ]
        for pat in set_name_patterns_hi:
            m = re.match(pat, lowered)
            if m:
                extracted_name = m.groupdict().get("name", "").strip()
                if extracted_name and extracted_name.lower() not in {"hai", "kya", "batao", "karo"}:
                    return IntentResult(
                        intent=IntentType.SET_USER_NAME,
                        parameters={"name": extracted_name.title(), "is_hindi": True},
                        confidence=0.95,
                        raw_text=cleaned,
                    )

        # English patterns
        set_name_patterns_en = [
            r"^(?:my\s+name\s+is|call\s+me|set\s+my\s+name\s+to|change\s+my\s+name\s+to|i\s+am)\s+(?P<name>[a-zA-Z\u0900-\u097F]+)$",
            r"^name\s+is\s+(?P<name>[a-zA-Z\u0900-\u097F]+)$",
        ]
        for pat in set_name_patterns_en:
            m = re.match(pat, lowered)
            if m:
                extracted_name = m.groupdict().get("name", "").strip()
                if extracted_name and extracted_name.lower() not in {"what", "who", "listening", "morphix"}:
                    return IntentResult(
                        intent=IntentType.SET_USER_NAME,
                        parameters={"name": extracted_name.title(), "is_hindi": False},
                        confidence=0.95,
                        raw_text=cleaned,
                    )

        # 3. GET USER NAME INTENT
        get_name_patterns_hi = [
            r"^(?:mera\s+naam\s+kya\s+hai|main\s+kaun\s+hoon|kya\s+tum\s+mera\s+naam\s+jaanti\s+ho)$",
        ]
        for pat in get_name_patterns_hi:
            if re.search(pat, lowered):
                return IntentResult(
                    intent=IntentType.GET_USER_NAME,
                    parameters={"is_hindi": True},
                    confidence=0.95,
                    raw_text=cleaned,
                )

        get_name_patterns_en = [
            r"^(?:who\s+am\s+i|what\s+is\s+my\s+name|do\s+you\s+know\s+my\s+name|what['’]?s\s+my\s+name)$",
        ]
        for pat in get_name_patterns_en:
            if re.search(pat, lowered):
                return IntentResult(
                    intent=IntentType.GET_USER_NAME,
                    parameters={"is_hindi": False},
                    confidence=0.95,
                    raw_text=cleaned,
                )

        return None

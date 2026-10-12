"""
Morphix Desktop Companion Robot - Surveillance Voice Controls Intent Plugin
===========================================================================
File: src/intent/surveillance_plugin.py
Description:
    Provides voice controls for activating and deactivating real-time silent
    surveillance guard mode and querying security reports.
"""

import logging
import re
from typing import Any, Dict, Optional

from src.config.profile_manager import get_primary_user
from src.intent.intent_classifier import IntentResult, IntentType
from src.vision.camera_manager import CameraManager

logger = logging.getLogger("Morphix.SurveillancePlugin")


class SurveillanceService:
    """
    Handles surveillance business logic and formats voice responses.
    """
    

    _instance: Optional["SurveillanceService"] = None

    @classmethod
    def get_instance(cls) -> "SurveillanceService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def start_surveillance(self, is_hindi: bool = False) -> str:
        """
        Activates silent surveillance guard mode and sets Sentry eyes.

        :param is_hindi: True for Hindi phrasing.
        :return: Spoken response string.
        """
        cam = CameraManager.get_instance()
        cam.start_surveillance()

        if is_hindi:
            return "Surveillance mode chalu ho gaya hai. Main aapke room par nazar rakh rahi hoon."
        return "Surveillance mode activated. Guarding your room."

    def stop_surveillance(self, is_hindi: bool = False) -> str:
        """
        Deactivates surveillance, restores normal eyes, and reports intruder detections.

        :param is_hindi: True for Hindi phrasing.
        :return: Spoken response string.
        """
        cam = CameraManager.get_instance()
        report = cam.stop_surveillance()
        intruders_count = report.get("intruders_captured", 0)
        user_name = get_primary_user()

        if intruders_count > 0:
            if is_hindi:
                event_str = "anokhi activity detect hui thi" if intruders_count == 1 else "anokhi activities detect hui thi"
                return f"Welcome back {user_name}! Surveillance mode band ho gaya hai. Dhyan dein: {intruders_count} {event_str}."
            event_str = "event was" if intruders_count == 1 else "events were"
            return f"Welcome back {user_name}! Surveillance mode deactivated. Note: {intruders_count} potential intruder {event_str} captured."
        else:
            if is_hindi:
                return f"Welcome back {user_name}! Surveillance mode band kar diya gaya hai. Koi anjaan vyakti nahi aaya."
            return f"Welcome back {user_name}! Surveillance mode deactivated."

    def get_status(self, is_hindi: bool = False) -> str:
        """
        Returns spoken summary of surveillance status.

        :param is_hindi: True for Hindi phrasing.
        :return: Spoken response string.
        """
        cam = CameraManager.get_instance()
        is_active = cam.is_surveillance_active()
        recent_events = cam.get_recent_events(limit=5)
        intruders_in_recent = sum(1 for e in recent_events if e.get("event_type") == "UNKNOWN")

        if is_active:
            current_count = cam.get_session_intruder_count()
            if is_hindi:
                return f"Surveillance mode active hai. Abhi tak {current_count} anjaan activity detect hui hai."
            return f"Surveillance mode is active. {current_count} intruder detections recorded this session."
        else:
            if is_hindi:
                return f"Surveillance mode abhi inactive hai. Pichhle record me {intruders_in_recent} unknown events the."
            return f"Surveillance mode is currently inactive. Recent log shows {intruders_in_recent} unknown intruder events."

    def delete_surveillance_captures(self, is_hindi: bool = False) -> str:
        """
        Purges all surveillance snapshots and clears the event logs.

        :param is_hindi: True for Hindi phrasing.
        :return: Spoken response confirmation string.
        """
        cam = CameraManager.get_instance()
        success, deleted_count, msg = cam.clear_surveillance_captures()

        if is_hindi:
            if deleted_count > 0:
                return f"Surveillance captures aur {deleted_count} photos delete kar di gayi hain."
            return "Surveillance folder me koi photos nahi hain. Event log reset kar diya gaya hai."

        if deleted_count > 0:
            return "All surveillance captures and event logs have been deleted successfully."
        return "No surveillance captures found. Event logs are clear."


class SurveillanceIntentPlugin:
    """
    Regex-based Intent Plugin for voice surveillance commands.
    """

    def __init__(self) -> None:
        self.service = SurveillanceService.get_instance()

    def match(self, text: str) -> Optional[IntentResult]:
        """
        Matches user commands against surveillance intent patterns.

        :param text: Raw spoken input string.
        :return: IntentResult if matched, None otherwise.
        """
        if not text or not isinstance(text, str):
            return None

        cleaned = text.strip()
        lowered = cleaned.lower()
        is_hindi = bool(re.search(r'\b(karo|kar do|mera|kya|hai|aa gaya|wapas|bulao|chehra|rakhwali|kamre|chalu|band|tha|aaya|batao|saari|hata|hatao)\b', lowered))

        # 1. DELETE SURVEILLANCE CAPTURES / CLEAR PHOTOS
        delete_patterns = [
            r"\b(?:delete|clear|remove|purge)\s+(?:all\s+)?(?:the\s+)?(?:surveillance|intruder|captured)?\s*(?:photos|images|captures|pictures|snapshots)\b",
            r"\b(?:delete|clear|remove|purge)\s+(?:surveillance|intruder|captured)\s+(?:photos|images|captures|pictures|snapshots)?\b",
            r"\b(?:delete|clear)\s+(?:photos|images|captures|pictures|snapshots)\b",
            r"\b(?:surveillance|intruder)?\s*(?:photos|images|captures|snapshots|pictures)\s+(?:delete|clear|remove|hata|hatao)(?:\s+(?:karo|kar\s+do|kijiye))?\b",
            r"\b(?:saari\s+)?(?:surveillance|intruder)?\s*(?:photos|images|captures)\s+(?:delete|hata|clear)(?:\s+(?:karo|kar\s+do))?\b",
        ]
        for pat in delete_patterns:
            if re.search(pat, lowered):
                return IntentResult(
                    intent=IntentType.DELETE_SURVEILLANCE_CAPTURES,
                    parameters={"is_hindi": is_hindi},
                    confidence=0.98,
                    raw_text=cleaned,
                )

        # 2. STOP SURVEILLANCE / I AM BACK
        stop_patterns = [
            r"\b(?:i\s+am|i'm)\s+back\b",
            r"\b(?:main\s+)?wapas\s+aa\s+gaya(?:\s+hoon)?\b",
            r"\b(?:stop|deactivate|disable|turn\s+off|band|off)\s+(?:the\s+)?(?:surveillance|guard|security)(?:\s+mode)?\b",
            r"\b(?:surveillance|guard|security)(?:\s+mode)?\s+(?:band|off|stop|deactivate)\b",
            r"\bstop\s+guarding\b",
            r"\bguard\s+mode\s+band\b",
        ]
        for pat in stop_patterns:
            if re.search(pat, lowered):
                return IntentResult(
                    intent=IntentType.STOP_SURVEILLANCE,
                    parameters={"is_hindi": is_hindi},
                    confidence=0.96,
                    raw_text=cleaned,
                )

        # 3. GET SURVEILLANCE STATUS / INTRUDER CHECK
        status_patterns = [
            r"\b(?:check\s+)?(?:the\s+)?(?:surveillance|security|guard)\s+(?:status|report)\b",
            r"\b(?:did\s+anyone|has\s+anyone)\s+(?:enter|come\s+in)(?:\s+my\s+room)?\b",
            r"\b(?:any\s+)?intruders(?:\s+detected)?\b",
            r"\b(?:is\s+)?surveillance\s+active\b",
            r"\b(?:kya\s+)?koi\s+(?:aaya\s+tha|kamre\s+me\s+aaya|room\s+me\s+aaya)(?:\s+tha)?\b",
        ]
        for pat in status_patterns:
            if re.search(pat, lowered):
                return IntentResult(
                    intent=IntentType.GET_SURVEILLANCE_STATUS,
                    parameters={"is_hindi": is_hindi},
                    confidence=0.94,
                    raw_text=cleaned,
                )

        # 4. START SURVEILLANCE
        start_patterns = [
            r"\b(?:activate|start|enable|turn\s+on|chalu)\s+(?:the\s+)?(?:surveillance|guard|security)(?:\s+mode)?\b",
            r"\b(?:surveillance|guard|security)(?:\s+mode)?\s+(?:chalu|on|activate|start)\b",
            r"\b(?:watch|guard)\s+(?:my\s+)?room\b",
            r"\b(?:surveillance|guard|security)\s+mode\s+on\b",
            r"\bstart\s+guarding\b",
            r"\b(?:room|kamre)\s+ki\s+surveillance\b",
            r"\b(?:room|kamre)\s+(?:pe\s+nazar|ki\s+rakhwali)\b",
        ]
        for pat in start_patterns:
            if re.search(pat, lowered):
                return IntentResult(
                    intent=IntentType.START_SURVEILLANCE,
                    parameters={"is_hindi": is_hindi},
                    confidence=0.96,
                    raw_text=cleaned,
                )

        return None

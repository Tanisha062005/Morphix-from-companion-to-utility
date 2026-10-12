"""
Morphix Desktop Companion Robot - Plugin-Based Offline Intent Classifier
========================================================================
File: src/intent/intent_classifier.py
Description:
    Provides a rule-based, offline Intent Classifier using a Plugin Registry architecture.
    Does NOT use online APIs, Gemini, or heavy ML models. Fast, low CPU/RAM footprint.
    Designed for Raspberry Pi and desktop execution.
"""

from dataclasses import dataclass, field
from enum import Enum
import logging
import re
import sys
from typing import Any, Dict, List, Optional

# Setup logger for IntentClassifier
logger = logging.getLogger("Morphix.Intent")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [Intent]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class IntentType(str, Enum):
    """Supported intent types for Morphix."""
    ADD_TODO = "ADD_TODO"
    SHOW_TODO = "SHOW_TODO"
    DELETE_TODO = "DELETE_TODO"
    COMPLETE_TODO = "COMPLETE_TODO"
    CLEAR_TODO = "CLEAR_TODO"
    ADD_REMINDER = "ADD_REMINDER"
    SHOW_REMINDERS = "SHOW_REMINDERS"
    DELETE_REMINDER = "DELETE_REMINDER"
    GET_WEATHER = "GET_WEATHER"
    PLAY_MUSIC = "PLAY_MUSIC"
    PAUSE_MUSIC = "PAUSE_MUSIC"
    RESUME_MUSIC = "RESUME_MUSIC"
    STOP_MUSIC = "STOP_MUSIC"
    NEXT_MUSIC = "NEXT_MUSIC"
    SET_VOLUME = "SET_VOLUME"
    INCREASE_VOLUME = "INCREASE_VOLUME"
    DECREASE_VOLUME = "DECREASE_VOLUME"
    MUTE_VOLUME = "MUTE_VOLUME"
    UNMUTE_VOLUME = "UNMUTE_VOLUME"
    GET_SYSTEM_HEALTH = "GET_SYSTEM_HEALTH"
    GET_FESTIVAL = "GET_FESTIVAL"
    SET_USER_NAME = "SET_USER_NAME"
    GET_USER_NAME = "GET_USER_NAME"
    ENROLL_FACE = "ENROLL_FACE"
    START_SURVEILLANCE = "START_SURVEILLANCE"
    STOP_SURVEILLANCE = "STOP_SURVEILLANCE"
    GET_SURVEILLANCE_STATUS = "GET_SURVEILLANCE_STATUS"
    DELETE_SURVEILLANCE_CAPTURES = "DELETE_SURVEILLANCE_CAPTURES"
    PLAY_RPS = "PLAY_RPS"
    UNKNOWN = "UNKNOWN"


@dataclass
class IntentResult:
    """Represents the structured output of intent classification."""
    intent: IntentType
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    raw_text: str = ""


class TodoIntentPlugin:
    """
    Plugin handling To-Do list related intents:
    ADD_TODO, SHOW_TODO, COMPLETE_TODO, DELETE_TODO, CLEAR_TODO.
    """

    def match(self, text: str) -> Optional[IntentResult]:
        lowercased = text.lower().strip()

        # 1. CLEAR_TODO
        clear_patterns = [
            r"clear (?:my )?(?:to-?do|todo|tudu|tasks?)(?: list)?$",
            r"clear (?:todo|tudu)$",
            r"empty (?:my )?(?:to-?do|todo|tudu|tasks?)(?: list)?$",
            r"delete all (?:tasks|to-?dos?|tudus?)$",
            r"remove all (?:tasks|to-?dos?|tudus?)$",
            r"^saare\s+(?:todo|tudu|tasks?)\s+(?:delete|hata|clear)(?:\s+(?:kar\s+)?do|\s+karo)?$"
        ]
        for pat in clear_patterns:
            if re.search(pat, lowercased):
                return IntentResult(intent=IntentType.CLEAR_TODO, confidence=1.0, raw_text=text)

        # 2. COMPLETE_TODO
        complete_patterns = [
            r"i completed (?P<task>.+)",
            r"mark (?P<task>.+) as done",
            r"complete (?P<task>.+)",
            r"finish (?P<task>.+)",
            r"completed (?P<task>.+)"
        ]
        for pat in complete_patterns:
            m = re.match(pat, lowercased)
            if m:
                task_str = m.group("task").strip()
                task_str = re.sub(r"\s+from (?:my )?(?:to-?do|todo|tudu|tasks?)(?: list)?$", "", task_str).strip()
                return IntentResult(
                    intent=IntentType.COMPLETE_TODO,
                    parameters={"task": task_str},
                    confidence=0.95,
                    raw_text=text
                )

        # 3. DELETE_TODO
        delete_patterns = [
            r"delete task (?P<task>.+)",
            r"delete (?P<task>.+) from (?:my )?(?:to-?do|todo|tudu|tasks?)(?: list)?",
            r"delete (?P<task>.+)",
            r"remove task (?P<task>.+)",
            r"remove (?P<task>.+) from (?:my )?(?:to-?do|todo|tudu|tasks?)(?: list)?",
            r"remove (?P<task>.+)",
            r"^(?P<task>.+?)\s+(?:todo|tudu|list)\s+(?:se\s+)?(?:delete|hata|remove)(?:\s+(?:kar\s+)?do|\s*karo)?$"
        ]
        for pat in delete_patterns:
            m = re.match(pat, lowercased)
            if m:
                task_str = m.group("task").strip()
                if not any(k in task_str for k in ["reminder", "reminders"]):
                    return IntentResult(
                        intent=IntentType.DELETE_TODO,
                        parameters={"task": task_str},
                        confidence=0.90,
                        raw_text=text
                    )

        # 4. Explicit ADD_TODO
        add_patterns_explicit = [
            r"add (?P<task>.+) to (?:my )?(?:to-?do|todo|tudu)(?:\s+list)?",
            r"add (?P<task>.+) in (?:my )?(?:to-?do|todo|tudu)(?:\s+list)?",
            r"add task (?P<task>.+)",
            r"(?:to-?do|todo|tudu|list) mein (?P<task>.+) (?:daal do|add karo|likh do)",
            r"^(?:to-?do|todo|tudu|list)\s+(?:mein|me)\s+(?P<task>.+?)\s+(?:daal do|add karo|likh do|add kar do|karo)$",
            r"(?:टूडू|टुडू)\s+में\s+(?P<task>.+)\s+(?:जोड़ो|डालो)"
        ]
        _todo_suffix_re = re.compile(
            r"\s+(?:in|to|into|on)\s+(?:my\s+)?(?:to-?do|todo|tudu|tasks?)(?:\s+list)?$",
            re.IGNORECASE
        )
        for pat in add_patterns_explicit:
            m = re.match(pat, lowercased)
            if m:
                task_str = m.group("task").strip()
                task_str = _todo_suffix_re.sub("", task_str).strip()
                if not any(k in task_str for k in ["remind", "reminder", "reminders"]):
                    return IntentResult(
                        intent=IntentType.ADD_TODO,
                        parameters={"task": task_str},
                        confidence=0.95,
                        raw_text=text
                    )

        # 5. SHOW_TODO (Hinglish & English)
        show_patterns = [
            r".*(?:show|dikha|bata).*(?:to-?do|todo|tudu|list).*",
            r".*(?:meri|mera|mere|my).*(?:to-?do|todo|tudu).*",
            r"show (?:my )?(?:to-?do|todo|tudu|tasks?)(?: list)?$",
            r"show me my (?:to-?do|todo|tudu|tasks?)(?: list)?$",
            r"show (?:to-?do|todo|tudu) list$",
            r"^what(?:'s| is)? on my (?:to-?do|todo|tudu|tasks?)(?: list)?$",
            r"^my (?:to-?do|todo|tudu|tasks?) list$",
            r"what are my (?:tasks|to-?dos?|tudus?)$",
            r"read (?:my )?(?:to-?do|todo|tudu|tasks?)(?: list)?$",
            r"read (?:my )?(?:todo|tudu)$",
            r"list (?:my )?tasks$",
            r"get (?:my )?tasks$",
            r"show (?:todo|tudu)$",
            r"read (?:todo|tudu)$",
            r"(?:mera|mere|meri)\s+(?:todo|tudu)\s+dikhao$",
            r"^(?:mera|mere|meri)?\s*(?:todo|tudu)\s+(?:dikhao|dikha|batao|bata|kya\s+kya\s+hai|kya hai).*$",
            r".*(?:to-?do|todo|tudu)\s+(?:list\s+)?(?:dikha|dikhao|bata|batao).*",
            r".*(?:display|screen)\s+(?:per|par|pe)\s+(?:to-?do|todo|tudu|tasks?).*"
        ]
        if not re.match(r"^(?:add|insert|likh|daal)\b", lowercased):
            for pat in show_patterns:
                if re.search(pat, lowercased):
                    return IntentResult(intent=IntentType.SHOW_TODO, confidence=1.0, raw_text=text)

        # 6. Generic ADD_TODO (fallback)
        generic_add_patterns = [
            r"add (?P<task>.+)"
        ]
        for pat in generic_add_patterns:
            m = re.match(pat, lowercased)
            if m:
                task_str = m.group("task").strip()
                task_str = _todo_suffix_re.sub("", task_str).strip()
                if not any(k in task_str for k in ["remind", "reminder", "reminders"]):
                    return IntentResult(
                        intent=IntentType.ADD_TODO,
                        parameters={"task": task_str},
                        confidence=0.85,
                        raw_text=text
                    )

        return None

        return None


class ReminderIntentPlugin:
    """
    Plugin handling Reminder related intents:
    ADD_REMINDER, SHOW_REMINDERS, DELETE_REMINDER.
    """

    def match(self, text: str) -> Optional[IntentResult]:
        lowercased = text.lower().strip()

        # 1. SHOW_REMINDERS
        show_rem_patterns = [
            r"^show (?:me )?(?:my )?(?:upcoming )?reminders?\??$",
            r"^what are my (?:upcoming )?reminders\??$",
            r"^read (?:me )?(?:my )?reminders?\??$",
            r"^list (?:my )?reminders?\??$",
            r"^show (?:my )?schedule\??$",
            r"^check (?:my |the )?reminders?(?:\s+for\s+me)?(?:\s+right\s+now)?\??$",
            r"^(?:do\s+i\s+have\s+)?any\s+(?:kind\s+of\s+)?reminders(?:\s+for\s+me)?(?:\s+right\s+now)?\??$",
            r"^do\s+i\s+have\s+(?:any\s+)?(?:kind\s+of\s+)?reminders?(?:\s+for\s+me)?(?:\s+right\s+now)?\??$",
            r"^did i have any reminders\??$",
            r"^(?:mere|mera)\s+reminders?\s+(?:dikhao|batao|check\s+karo)\??$",
            r"^(?:kya\s+)?(?:mera\s+|mere\s+)?(?:koi\s+)?reminders?\s+(?:hai|batao|dikhao)(?:\s+kya)?\??$",
            r"^(?:kya\s+)?(?:koi\s+)?(?:mera\s+|mere\s+)?reminders?\s+(?:hai|batao|dikhao)(?:\s+kya)?\??$",
            r"^(?:kya\s+)?koi\s+reminder\s+hai\??$"
        ]
        for pat in show_rem_patterns:
            if re.search(pat, lowercased):
                return IntentResult(intent=IntentType.SHOW_REMINDERS, confidence=1.0, raw_text=text)

        # 2. DELETE_REMINDER
        del_rem_patterns = [
            r"delete reminder (?P<target>.+)",
            r"remove reminder (?P<target>.+)"
        ]
        for pat in del_rem_patterns:
            m = re.match(pat, lowercased)
            if m:
                target_str = m.group("target").strip()
                return IntentResult(
                    intent=IntentType.DELETE_REMINDER,
                    parameters={"target": target_str},
                    confidence=0.95,
                    raw_text=text
                )

        # 3. ADD_REMINDER
        # Check Hindi/Hinglish remind patterns first
        p_hindi1 = r"^(?:mujhe|mera|mere|मुझे)\s+(?P<time_expr>\d+\s*(?:minute|min|seconds|sec|hour|hr|minutes|hours|second|secs|hours|days|day|दिन|घंटे|मिनट|सेकंड|सेकण्ड|आर)(?:\s*(?:baad|mein|me|in|after|at|बाद|में))?)\s+(?P<text>.+?)\s+(?:yaad dilana|remind karna|yaad dilaye|याद दिलाना)(?:\s*(?:baad|mein|me|बाद|में))?$"
        p_hindi2 = r"^(?:mujhe|mera|mere|मुझे)\s+(?P<text>.+?)\s+(?P<time_expr>\d+\s*(?:minute|min|seconds|sec|hour|hr|minutes|hours|second|secs|hours|days|day|दिन|घंटे|मिनट|सेकंड|सेकण्ड|आर)(?:\s*(?:baad|mein|me|in|after|at|बाद|में))?)\s+(?:yaad dilana|remind karna|yaad dilaye|याद दिलाना)$"
        p_hindi3 = r"^(?:mujhe|mera|mere|मुझे)\s+(?P<text>.+?)\s+(?:yaad dilana|remind karna|yaad dilaye|याद दिलाना)\s+(?P<time_expr>.+)$"
        
        # User requested specific patterns
        p_req_hi = r"^(?:mujhe|mera)\s+(?P<text>.+?)\s+yaad dilana$"
        p_req_deva = r"^मुझे\s+(?P<text>.+?)\s+याद दिलाना$"
        p_hindi_fallback = r"^(?:mujhe|mera|mere|मुझे)\s+(?P<text>.+?)\s+(?:yaad dilana|remind karna|yaad dilaye|याद दिलाना)$"

        for pat in [p_hindi1, p_hindi2, p_hindi3, p_req_hi, p_req_deva, p_hindi_fallback]:
            m = re.match(pat, lowercased)
            if m:
                groups = m.groupdict()
                text_param = groups.get("text", "").strip() if groups.get("text") else ""
                time_param = groups.get("time_expr", "").strip() if groups.get("time_expr") else ""
                return IntentResult(
                    intent=IntentType.ADD_REMINDER,
                    parameters={"text": text_param if text_param else lowercased, "time_expr": time_param},
                    confidence=0.95,
                    raw_text=text
                )

        remind_patterns = [
            r"remind me to (?P<text>.+) (?:in|after) (?P<time_expr>.+)", # User requested English pattern
            r"remind me to (?P<text>.+) (at|in|after|today at|tomorrow at) (?P<time_expr>.+)",
            r"remind me (at|in|after|today at|tomorrow at) (?P<time_expr>.+) to (?P<text>.+)",
            r"remind me to (?P<text>.+)",
            r"remind me (at|in|after|today at|tomorrow at) (?P<time_expr>.+)",
            r"set a reminder for (?P<text>.+) at (?P<time_expr>.+)",
            r"set a reminder to (?P<text>.+) at (?P<time_expr>.+)"
        ]
        for pat in remind_patterns:
            m = re.match(pat, lowercased)
            if m:
                groups = m.groupdict()
                text_param = groups.get("text", "").strip() if groups.get("text") else ""
                time_param = groups.get("time_expr", "").strip() if groups.get("time_expr") else ""

                if not time_param and " at " in lowercased:
                    parts = lowercased.split(" at ", 1)
                    text_param = parts[0].replace("remind me to", "").replace("remind me", "").strip()
                    time_param = parts[1].strip()
                elif not time_param and " in " in lowercased:
                    parts = lowercased.split(" in ", 1)
                    text_param = parts[0].replace("remind me to", "").replace("remind me", "").strip()
                    time_param = "in " + parts[1].strip()
                elif not time_param and " after " in lowercased:
                    parts = lowercased.split(" after ", 1)
                    text_param = parts[0].replace("remind me to", "").replace("remind me", "").strip()
                    time_param = "after " + parts[1].strip()

                return IntentResult(
                    intent=IntentType.ADD_REMINDER,
                    parameters={"text": text_param if text_param else lowercased, "time_expr": time_param},
                    confidence=0.95,
                    raw_text=text
                )

        return None


class IntentClassifier:
    """
    Plugin-based Intent Classifier for Morphix.
    Enables future modules (Weather, Camera, Battery, Gemini, etc.) to register custom intent handlers.
    """

    def __init__(self) -> None:
        self._plugins: List[Any] = []
        # Register default Morphix Phase 1 & 2 plugins
        try:
            from src.intent.todo_plugin import TodoIntentPlugin as ExtTodoPlugin
            self.register_plugin(ExtTodoPlugin())
        except Exception:
            self.register_plugin(TodoIntentPlugin())

        try:
            from src.intent.reminder_plugin import ReminderIntentPlugin as ExtReminderPlugin
            self.register_plugin(ExtReminderPlugin())
        except Exception:
            self.register_plugin(ReminderIntentPlugin())

        try:
            from src.intent.weather_plugin import WeatherIntentPlugin
            self.register_plugin(WeatherIntentPlugin())
        except Exception as e:
            logger.warning("Could not load WeatherIntentPlugin: %s", str(e))

        try:
            from src.intent.profile_plugin import ProfileIntentPlugin
            self.register_plugin(ProfileIntentPlugin())
        except Exception as e:
            logger.warning("Could not load ProfileIntentPlugin: %s", str(e))

        try:
            from src.intent.surveillance_plugin import SurveillanceIntentPlugin
            self.register_plugin(SurveillanceIntentPlugin())
        except Exception as e:
            logger.warning("Could not load SurveillanceIntentPlugin: %s", str(e))

        try:
            from src.intent.music_plugin import MusicIntentPlugin
            self.register_plugin(MusicIntentPlugin())
        except Exception as e:
            logger.warning("Could not load MusicIntentPlugin: %s", str(e))

        try:
            from src.intent.volume_plugin import VolumeIntentPlugin
            self.register_plugin(VolumeIntentPlugin())
        except Exception as e:
            logger.warning("Could not load VolumeIntentPlugin: %s", str(e))

        try:
            from src.intent.system_health_plugin import SystemHealthIntentPlugin
            self.register_plugin(SystemHealthIntentPlugin())
        except Exception as e:
            logger.warning("Could not load SystemHealthIntentPlugin: %s", str(e))

        try:
            from src.intent.festival_plugin import FestivalIntentPlugin
            self.register_plugin(FestivalIntentPlugin())
        except Exception as e:
            logger.warning("Could not load FestivalIntentPlugin: %s", str(e))

        try:
            from src.intent.rps_game_plugin import RPSGamePlugin
            self.register_plugin(RPSGamePlugin())
        except Exception as e:
            logger.warning("Could not load RPSGamePlugin: %s", str(e))

    def register_plugin(self, plugin: Any) -> None:
        """
        Registers a new intent plugin handler.

        :param plugin: Intent plugin object providing a match(text: str) method.
        """
        self._plugins.append(plugin)
        logger.info("Registered intent plugin: %s", plugin.__class__.__name__)

    def _intercept_hardware_and_surveillance(self, text: str) -> Optional[IntentResult]:
        """
        Strict Fast-Path Local Routing for Surveillance, Face Enrollment,
        and User Profile commands. Guarantees 100% offline local resolution
        and prevents hardware commands from ever falling through to Groq/LLM.
        """
        import re
        lowered = text.lower().strip()
        is_hindi = bool(re.search(r'\b(karo|kar do|mera|kya|hai|aa gaya|wapas|bulao|chehra|rakhwali|kamre|chalu|band)\b', lowered))

        # 0. Rock-Paper-Scissors Fast-Path
        rps_keywords = [
            "rock paper scissors", "rock paper scissor", "stone paper scissors",
            "stone paper scissor", "play rps", "lets play rps", "khelo rock paper",
            "khelte hain rock paper", "khelte hain stone paper", "play rock paper"
        ]
        if any(kw in lowered for kw in rps_keywords):
            return IntentResult(
                intent=IntentType.PLAY_RPS,
                parameters={"is_hindi": is_hindi},
                confidence=0.99,
                raw_text=text,
            )

        # 1. Face Enrollment
        enroll_keywords = [
            "register my face", "enroll my face", "face scan", "scan my face",
            "mera face", "chehra scan", "capture my face", "scan face", "register face",
            "enroll face", "mera chehra", "add my face"
        ]
        if any(kw in lowered for kw in enroll_keywords):
            return IntentResult(
                intent=IntentType.ENROLL_FACE,
                parameters={"is_hindi": is_hindi},
                confidence=0.99,
                raw_text=text,
            )

        # 2. Profile / Name Identification
        set_name_match = re.search(r'\b(?:my name is|call me|mera naam)\s+([a-zA-Z\u0900-\u097F]+)', lowered)
        if set_name_match:
            name = set_name_match.group(1).title()
            if name.lower() not in {"what", "who", "kya", "hai", "is", "listening", "guard", "mode"}:
                return IntentResult(
                    intent=IntentType.SET_USER_NAME,
                    parameters={"name": name, "is_hindi": is_hindi},
                    confidence=0.99,
                    raw_text=text,
                )

        if any(phrase in lowered for phrase in ["who am i", "what is my name", "what's my name", "mera naam kya hai", "main kaun hoon"]):
            return IntentResult(
                intent=IntentType.GET_USER_NAME,
                parameters={"is_hindi": is_hindi},
                confidence=0.99,
                raw_text=text,
            )

        # 3. Surveillance Snapshot & Log Deletion
        delete_captures_keywords = [
            "delete surveillance images", "delete surveillance photos", "delete surveillance captures",
            "clear captured photos", "clear captures", "clear surveillance images", "clear surveillance photos",
            "delete intruder photos", "delete intruder images", "delete photos", "delete images",
            "delete captures", "remove surveillance photos", "remove surveillance captures",
            "surveillance photos delete", "surveillance captures delete", "photos delete", "images delete"
        ]
        if any(kw in lowered for kw in delete_captures_keywords) or (
            ("delete" in lowered or "clear" in lowered or "remove" in lowered or "hata" in lowered) and
            ("photo" in lowered or "photos" in lowered or "image" in lowered or "images" in lowered or "capture" in lowered or "captures" in lowered or "intruder" in lowered) and
            not any(t in lowered for t in ["todo", "task", "reminder", "reminders"])
        ):
            return IntentResult(
                intent=IntentType.DELETE_SURVEILLANCE_CAPTURES,
                parameters={"is_hindi": is_hindi},
                confidence=0.99,
                raw_text=text,
            )

        # 4. Surveillance & Guard Controls
        surveillance_keywords = [
            "surveillance", "guard mode", "guarding", "watch my room", "security mode", "rakhwali",
            "room ki surveillance", "room pe nazar"
        ]
        has_surveillance_kw = any(kw in lowered for kw in surveillance_keywords) or (
            ("guard" in lowered or "security" in lowered) and ("start" in lowered or "stop" in lowered or "mode" in lowered or "on" in lowered or "off" in lowered or "chalu" in lowered or "band" in lowered)
        )

        if has_surveillance_kw or "i am back" in lowered or "wapas aa gaya" in lowered or "i'm back" in lowered:
            # Stop surveillance
            if any(stop_kw in lowered for stop_kw in ["stop", "deactivate", "disable", "turn off", "band", "off", "back", "wapas", "cancel"]):
                return IntentResult(
                    intent=IntentType.STOP_SURVEILLANCE,
                    parameters={"is_hindi": is_hindi},
                    confidence=0.99,
                    raw_text=text,
                )
            # Surveillance status
            if any(stat_kw in lowered for stat_kw in ["status", "report", "check", "anyone", "intruder", "koi aaya"]):
                return IntentResult(
                    intent=IntentType.GET_SURVEILLANCE_STATUS,
                    parameters={"is_hindi": is_hindi},
                    confidence=0.99,
                    raw_text=text,
                )
            # Otherwise default to Start surveillance
            return IntentResult(
                intent=IntentType.START_SURVEILLANCE,
                parameters={"is_hindi": is_hindi},
                confidence=0.99,
                raw_text=text,
            )

        return None

    def classify(self, text: str) -> IntentResult:
        """
        Classifies input command text by evaluating registered intent plugins.

        :param text: Spoken command string.
        :return: IntentResult object containing intent, parameters, and confidence score.
        """
        if not text or not isinstance(text, str):
            return IntentResult(intent=IntentType.UNKNOWN, confidence=0.0, raw_text="")

        clean_text = text.strip()

        # Step 0: Fast-Path Local Hardware & Surveillance Interceptor
        fast_res = self._intercept_hardware_and_surveillance(clean_text)
        if fast_res is not None:
            logger.info("Fast-Path intercept: '%s' -> intent '%s' (Confidence: %.2f)", clean_text, fast_res.intent.value, fast_res.confidence)
            return fast_res

        for plugin in self._plugins:
            try:
                res = plugin.match(clean_text)
                if res and res.intent != IntentType.UNKNOWN:
                    logger.info("Command '%s' classified as intent '%s' (Confidence: %.2f)", clean_text, res.intent.value, res.confidence)
                    return res
            except Exception as e:
                logger.error("Error evaluating plugin %s: %s", plugin.__class__.__name__, str(e))

        logger.info("Command '%s' unclassified -> UNKNOWN", clean_text)
        return IntentResult(intent=IntentType.UNKNOWN, confidence=0.0, raw_text=clean_text)

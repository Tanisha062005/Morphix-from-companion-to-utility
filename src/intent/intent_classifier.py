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
            r"clear (my )?(to-?do|tasks?)( list)?$",
            r"delete all (tasks|to-?dos?)$",
            r"remove all (tasks|to-?dos?)$"
        ]
        for pat in clear_patterns:
            if re.search(pat, lowercased):
                return IntentResult(intent=IntentType.CLEAR_TODO, confidence=1.0, raw_text=text)

        # 2. SHOW_TODO
        show_patterns = [
            r"show (my )?(to-?do|tasks?)( list)?$",
            r"what are my (tasks|to-?dos?)$",
            r"read (my )?(to-?do|tasks?)( list)?$",
            r"list (my )?tasks$",
            r"get (my )?tasks$"
        ]
        for pat in show_patterns:
            if re.search(pat, lowercased):
                return IntentResult(intent=IntentType.SHOW_TODO, confidence=1.0, raw_text=text)

        # 3. COMPLETE_TODO
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
                task_str = re.sub(r"\s+from (my )?(to-?do|tasks?)( list)?$", "", task_str).strip()
                return IntentResult(
                    intent=IntentType.COMPLETE_TODO,
                    parameters={"task": task_str},
                    confidence=0.95,
                    raw_text=text
                )

        # 4. DELETE_TODO
        delete_patterns = [
            r"delete task (?P<task>.+)",
            r"delete (?P<task>.+) from (my )?(to-?do|tasks?)( list)?",
            r"delete (?P<task>.+)",
            r"remove task (?P<task>.+)",
            r"remove (?P<task>.+) from (my )?(to-?do|tasks?)( list)?",
            r"remove (?P<task>.+)"
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

        # 5. ADD_TODO
        add_patterns = [
            r"add (?P<task>.+) to (my )?(to-?do|tasks?)( list)?",
            r"add task (?P<task>.+)",
            r"add (?P<task>.+)"
        ]
        for pat in add_patterns:
            m = re.match(pat, lowercased)
            if m:
                task_str = m.group("task").strip()
                if not any(k in task_str for k in ["remind", "reminder", "reminders"]):
                    return IntentResult(
                        intent=IntentType.ADD_TODO,
                        parameters={"task": task_str},
                        confidence=0.90,
                        raw_text=text
                    )

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
            r"show (my )?reminders$",
            r"what are my reminders$",
            r"read (my )?reminders$",
            r"list (my )?reminders$"
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
        remind_patterns = [
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
        # Register default Morphix Phase 1 plugins
        self.register_plugin(TodoIntentPlugin())
        self.register_plugin(ReminderIntentPlugin())

    def register_plugin(self, plugin: Any) -> None:
        """
        Registers a new intent plugin handler.

        :param plugin: Intent plugin object providing a match(text: str) method.
        """
        self._plugins.append(plugin)
        logger.info("Registered intent plugin: %s", plugin.__class__.__name__)

    def classify(self, text: str) -> IntentResult:
        """
        Classifies input command text by evaluating registered intent plugins.

        :param text: Spoken command string.
        :return: IntentResult object containing intent, parameters, and confidence score.
        """
        if not text or not isinstance(text, str):
            return IntentResult(intent=IntentType.UNKNOWN, confidence=0.0, raw_text="")

        clean_text = text.strip()

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

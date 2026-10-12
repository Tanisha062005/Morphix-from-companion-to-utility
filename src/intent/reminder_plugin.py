"""
Morphix Desktop Companion Robot - Reminders Intent Plugin & Service
=====================================================================
File: src/intent/reminder_plugin.py
Description:
    Provides NLU intent matching for Reminder operations (ADD, SHOW, DELETE)
    and integrates with ReminderManager and the OLED display overlay system (EyeAnimator).
"""

from datetime import datetime
import logging
import re
import sys
import threading
from typing import Any, Dict, List, Optional

from src.display.eye_animator import get_active_display
from src.reminder.reminder_manager import ReminderManager

# Setup logger for Reminder Plugin
logger = logging.getLogger("Morphix.ReminderPlugin")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [ReminderPlugin]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class ReminderService:
    """
    Manages Reminder business operations and triggers display overlay cards on the OLED screen.
    """
    _instance: Optional["ReminderService"] = None
    _lock = threading.Lock()

    def __init__(self, reminder_manager: Optional[ReminderManager] = None) -> None:
        self.reminder_manager = reminder_manager or ReminderManager()

    @classmethod
    def get_instance(cls) -> "ReminderService":
        with cls._lock:
            if cls._instance is None:
                cls._instance = ReminderService()
            return cls._instance

    def show_reminders(self, is_hindi: bool = False, display: Optional[Any] = None) -> str:
        """
        Retrieves pending reminders, triggers on-screen Pygame overlay card for upcoming reminder,
        and returns formatted spoken response for TTS.
        """
        pending = self.reminder_manager.get_pending_reminders()

        if not pending:
            return "Koi pending reminder nahi hai." if is_hindi else "You have no pending reminders."

        # Build structured reminder list for the display overlay
        reminder_dicts = []
        for r in pending:
            rem_text = r.get("message") or r.get("text") or "Reminder"
            raw_trig = r.get("trigger_time", "")
            try:
                from datetime import datetime as _dt
                due_str = _dt.fromisoformat(raw_trig).strftime("%H:%M (%d %b)")
            except Exception:
                due_str = str(raw_trig)
            reminder_dicts.append({"text": rem_text, "due": due_str})

        # Trigger display overlay with full list \u2014 timeout covers TTS readback
        active_disp = display or get_active_display()
        if active_disp:
            try:
                if hasattr(active_disp, "show_reminders_overlay"):
                    active_disp.show_reminders_overlay(reminder_dicts, timeout=10.0)
                else:
                    # Legacy fallback for older thread instances
                    first = reminder_dicts[0] if reminder_dicts else {"text": "", "due": ""}
                    active_disp.show_reminder_overlay(
                        title=first["text"], due_time=first["due"], timeout=10.0
                    )
                logger.info("Sent %d reminders to display overlay.", len(reminder_dicts))
            except Exception as ex:
                logger.warning("Failed to render reminder overlay: %s", ex)

        rem_list_str = ", ".join(f"{r.get('message') or r.get('text')}" for r in pending)
        if is_hindi:
            return f"Aapke {len(pending)} pending reminders hain: {rem_list_str}."
        else:
            return f"You have {len(pending)} pending reminders: {rem_list_str}."

    def add_reminder(self, text: str, time_expr: str = "", is_hindi: bool = False, display: Optional[Any] = None) -> str:
        """Adds a reminder."""
        if not text or not text.strip():
            return "Kripya reminder text batayein." if is_hindi else "Please specify reminder details."

        try:
            res = self.reminder_manager.add_reminder(text.strip(), time_expr)
            trig_time = res.get("trigger_time", "")
            try:
                dt = datetime.fromisoformat(trig_time)
                trig_fmt = dt.strftime("%H:%M")
            except Exception:
                trig_fmt = time_expr or "soon"

            # Show reminder overlay for newly added reminder
            active_disp = display or get_active_display()
            if active_disp:
                try:
                    active_disp.show_reminder_overlay(title=text.strip(), due_time=trig_fmt, timeout=6.0)
                except Exception:
                    pass

            if is_hindi:
                return f"Reminder set kar diya: {text.strip()} ({trig_fmt} par)."
            else:
                return f"Reminder set for {text.strip()} at {trig_fmt}."
        except Exception as ex:
            logger.error("Error setting reminder: %s", ex)
            return "Reminder set karne mein error aaya." if is_hindi else "Failed to set reminder."

    def delete_reminder(self, target: str, is_hindi: bool = False) -> str:
        """Deletes a reminder matching target description."""
        deleted = self.reminder_manager.delete_reminder(target)
        if deleted:
            return f"Reminder delete kar diya: {target}." if is_hindi else f"Deleted reminder: {target}."
        return "Reminder nahi mila." if is_hindi else "Reminder not found."


class ReminderIntentPlugin:
    """
    NLU plugin handling Reminder operations:
    SHOW_REMINDERS, ADD_REMINDER, DELETE_REMINDER.
    """

    def __init__(self, service: Optional[ReminderService] = None) -> None:
        self.service = service or ReminderService.get_instance()

    def handle(self, intent_result: Any, display: Optional[Any] = None) -> str:
        """
        Executes the matching Reminder operation.
        When SHOW_REMINDERS triggers, renders Pygame card overlay and returns spoken text.
        """
        active_disp = display or get_active_display()
        intent = getattr(intent_result, "intent", intent_result)
        params = getattr(intent_result, "parameters", {}) if hasattr(intent_result, "parameters") else {}
        is_hindi = params.get("is_hindi", False)

        from src.intent.intent_classifier import IntentType
        if intent in (IntentType.SHOW_REMINDERS, "SHOW_REMINDERS"):
            return self.service.show_reminders(is_hindi=is_hindi, display=active_disp)
        elif intent in (IntentType.ADD_REMINDER, "ADD_REMINDER"):
            rem_text = params.get("text", "")
            time_expr = params.get("time_expr", "")
            return self.service.add_reminder(rem_text, time_expr, is_hindi=is_hindi, display=active_disp)
        elif intent in (IntentType.DELETE_REMINDER, "DELETE_REMINDER"):
            target_str = params.get("target", "")
            return self.service.delete_reminder(target_str, is_hindi=is_hindi)
        return "I'm not sure how to do that." if not is_hindi else "Mujhe samajh nahi aaya."

    def match(self, command: str) -> Optional[Any]:
        from src.intent.intent_classifier import IntentResult, IntentType

        if not command or not isinstance(command, str):
            return None

        text = command.strip()
        lowercased = text.lower()
        is_hindi = bool(re.search(r'\b(mujhe|mera|mere|yaad|dilana|dikhao|batao|kya|hai|dekhna|koi)\b', lowercased))

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
                return IntentResult(
                    intent=IntentType.SHOW_REMINDERS,
                    confidence=1.0,
                    parameters={"is_hindi": is_hindi},
                    raw_text=text
                )

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
                    confidence=0.95,
                    parameters={"target": target_str, "is_hindi": is_hindi},
                    raw_text=text
                )

        # 3. ADD_REMINDER
        p_hindi1 = r"^(?:mujhe|mera|mere|मुझे)\s+(?P<time_expr>\d+\s*(?:minute|min|seconds|sec|hour|hr|minutes|hours|second|secs|hours|days|day|दिन|घंटे|मिनट|सेकंड|सेकण्ड|आर)(?:\s*(?:baad|mein|me|in|after|at|बाद|में))?)\s+(?P<text>.+?)\s+(?:yaad dilana|remind karna|yaad dilaye|याद दिलाना)(?:\s*(?:baad|mein|me|बाद|में))?$"
        p_hindi2 = r"^(?:mujhe|mera|mere|मुझे)\s+(?P<text>.+?)\s+(?P<time_expr>\d+\s*(?:minute|min|seconds|sec|hour|hr|minutes|hours|second|secs|hours|days|day|दिन|घंटे|मिनट|सेकंड|सेकण्ड|आर)(?:\s*(?:baad|mein|me|in|after|at|बाद|में))?)\s+(?:yaad dilana|remind karna|yaad dilaye|याद दिलाना)$"
        p_hindi3 = r"^(?:mujhe|mera|mere|मुझे)\s+(?P<text>.+?)\s+(?:yaad dilana|remind karna|yaad dilaye|याद दिलाना)\s+(?P<time_expr>.+)$"
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
                    confidence=0.95,
                    parameters={"text": text_param if text_param else lowercased, "time_expr": time_param, "is_hindi": is_hindi},
                    raw_text=text
                )

        remind_patterns = [
            r"remind me to (?P<text>.+) (?:in|after) (?P<time_expr>.+)",
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
                    confidence=0.95,
                    parameters={"text": text_param if text_param else lowercased, "time_expr": time_param, "is_hindi": is_hindi},
                    raw_text=text
                )

        return None

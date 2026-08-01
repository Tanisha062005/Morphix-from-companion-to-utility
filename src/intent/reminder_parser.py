"""
Morphix Desktop Companion Robot - Reminder Intent Parser
=========================================================
File: src/intent/reminder_parser.py
Description:
    Rule-based parser that extracts reminder messages, raw time expressions,
    and relative delays (in seconds) from normalized speech inputs.
"""

import re
from typing import Any, Dict


class ReminderIntentParser:
    """
    Parses speech text to identify reminder intents and extract parameters.
    """

    @classmethod
    def parse(cls, text: str) -> Dict[str, Any]:
        """
        Parses text for SET_REMINDER intent and extracts message and timing parameters.

        :param text: Cleaned and normalized text input.
        :return: Dict containing intent and extracted fields.
        """
        if not text or not isinstance(text, str):
            return {"intent": "UNKNOWN"}

        cleaned = text.strip()

        # Reject if core keywords are missing or isolated
        if not any(kw in cleaned for kw in ["remind", "reminder"]):
            return {"intent": "UNKNOWN"}

        if cleaned in ["remind", "reminder"]:
            return {"intent": "UNKNOWN"}

        time_indicators = r"\b(after|in|at|on|today|tomorrow|next)\b"

        # Regex patterns for parsing reminder sentences
        p_to_time = rf"(?:remind me to|set a reminder to|set a reminder for|reminder to|reminder for)\s+(?P<message>.+?)\s+(?P<indicator>{time_indicators})\s+(?P<time_expr>.+)"
        p_time_to = rf"(?:remind me|set a reminder)\s+(?P<time_expr>(?:after|in|at|on|today|tomorrow|next)\s+.+?)\s+to\s+(?P<message>.+)"
        p_only_time = rf"(?:remind me|set a reminder|reminder)\s+(?P<time_expr>(?:after|in|at|on|today|tomorrow|next)\s+.+)"
        p_only_message = r"(?:remind me to|set a reminder to|set a reminder for|reminder to)\s+(?P<message>.+)"

        message = ""
        time_expr = ""

        m1 = re.match(p_to_time, cleaned, re.IGNORECASE)
        m2 = re.match(p_time_to, cleaned, re.IGNORECASE)
        m3 = re.match(p_only_time, cleaned, re.IGNORECASE)
        m4 = re.match(p_only_message, cleaned, re.IGNORECASE)

        if m1:
            message = m1.group("message").strip()
            time_expr = (m1.group("indicator") + " " + m1.group("time_expr")).strip()
        elif m2:
            message = m2.group("message").strip()
            time_expr = m2.group("time_expr").strip()
        elif m3:
            time_expr = m3.group("time_expr").strip()
        elif m4:
            message = m4.group("message").strip()
        else:
            # Fallback for generic patterns like: "remind tomorrow 5 pm"
            m_ind = re.search(time_indicators, cleaned)
            if m_ind:
                idx = m_ind.start()
                message_part = cleaned[:idx].replace("remind me to", "").replace("remind me", "").replace("remind", "").replace("set a reminder", "").replace("reminder", "").strip()
                time_part = cleaned[idx:].strip()
                message = message_part
                time_expr = time_part
            else:
                message = cleaned.replace("remind me to", "").replace("remind me", "").replace("remind", "").replace("set a reminder", "").strip()

        message = message.strip()
        time_expr = time_expr.strip()

        if not message and not time_expr:
            return {"intent": "UNKNOWN"}

        # Calculate relative delay (in seconds) if applicable
        delay = None
        if time_expr:
            seconds = 0
            minutes = 0
            hours = 0

            m_sec = re.search(r"(\d+)\s*(second|sec|seconds)", time_expr)
            if m_sec:
                seconds = int(m_sec.group(1))

            m_min = re.search(r"(\d+)\s*(minute|min|minutes)", time_expr)
            if m_min:
                minutes = int(m_min.group(1))

            m_hr = re.search(r"(\d+)\s*(hour|hr|hours)", time_expr)
            if m_hr:
                hours = int(m_hr.group(1))

            is_relative = (time_expr.startswith("in") or time_expr.startswith("after"))
            if is_relative and (seconds > 0 or minutes > 0 or hours > 0):
                delay = seconds + (minutes * 60) + (hours * 3600)

        result = {
            "intent": "SET_REMINDER",
            "message": message,
        }
        if delay is not None:
            result["delay"] = delay
        if time_expr:
            result["time_expr"] = time_expr

        return result

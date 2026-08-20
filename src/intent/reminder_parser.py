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
        Supports English, Hindi, and Hinglish text.

        :param text: Cleaned and normalized text input.
        :return: Dict containing intent and extracted fields.
        """
        if not text or not isinstance(text, str):
            return {"intent": "UNKNOWN"}

        # Normalize Hindi/Devanagari digits to standard Latin ASCII digits
        def normalize_digits(t: str) -> str:
            devanagari_to_latin = {
                '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
                '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'
            }
            for d, l in devanagari_to_latin.items():
                t = t.replace(d, l)
            return t

        cleaned = normalize_digits(text.strip())

        # Reject if core keywords are missing or isolated
        keywords = ["remind", "reminder", "yaad dilana", "yaad", "dilana", "याद", "दिलाना"]
        if not any(kw in cleaned.lower() for kw in keywords):
            return {"intent": "UNKNOWN"}

        if cleaned.lower() in ["remind", "reminder", "yaad", "याद"]:
            return {"intent": "UNKNOWN"}

        time_indicators = r"\b(after|in|at|on|today|tomorrow|next)\b"

        # Regex patterns for parsing English reminder sentences
        p_to_time = rf"(?:remind me to|set a reminder to|set a reminder for|reminder to|reminder for)\s+(?P<message>.+?)\s+(?P<indicator>{time_indicators})\s+(?P<time_expr>.+)"
        p_time_to = rf"(?:remind me|set a reminder)\s+(?P<time_expr>(?:after|in|at|on|today|tomorrow|next)\s+.+?)\s+to\s+(?P<message>.+)"
        p_only_time = rf"(?:remind me|set a reminder|reminder)\s+(?P<time_expr>(?:after|in|at|on|today|tomorrow|next)\s+.+)"
        p_only_message = r"(?:remind me to|set a reminder to|set a reminder for|reminder to)\s+(?P<message>.+)"

        # Regex patterns for Hindi/Hinglish reminders
        p_hindi1 = r"^(?:mujhe|mera|mere|मुझे)\s+(?P<time_expr>\d+\s*(?:minute|min|seconds|sec|hour|hr|minutes|hours|second|secs|hours|days|day|दिन|घंटे|मिनट|सेकंड|सेकण्ड|आर)(?:\s*(?:baad|mein|me|in|after|at|बाद|में))?)\s+(?P<text>.+?)\s+(?:yaad dilana|remind karna|yaad dilaye|याद दिलाना)(?:\s*(?:baad|mein|me|बाद|में))?$"
        p_hindi2 = r"^(?:mujhe|mera|mere|मुझे)\s+(?P<text>.+?)\s+(?P<time_expr>\d+\s*(?:minute|min|seconds|sec|hour|hr|minutes|hours|second|secs|hours|days|day|दिन|घंटे|मिनट|सेकंड|सेकण्ड|आर)(?:\s*(?:baad|mein|me|in|after|at|बाद|में))?)\s+(?:yaad dilana|remind karna|yaad dilaye|याद दिलाना)$"
        p_hindi3 = r"^(?:mujhe|mera|mere|मुझे)\s+(?P<text>.+?)\s+(?:yaad dilana|remind karna|yaad dilaye|याद दिलाना)\s+(?P<time_expr>.+)$"
        p_hindi_fallback = r"^(?:mujhe|mera|mere|मुझे)\s+(?P<text>.+?)\s+(?:yaad dilana|remind karna|yaad dilaye|याद दिलाना)$"

        message = ""
        time_expr = ""

        # Match Hindi/Hinglish patterns first
        mh1 = re.match(p_hindi1, cleaned, re.IGNORECASE)
        mh2 = re.match(p_hindi2, cleaned, re.IGNORECASE)
        mh3 = re.match(p_hindi3, cleaned, re.IGNORECASE)
        mh_fb = re.match(p_hindi_fallback, cleaned, re.IGNORECASE)

        if mh1:
            message = mh1.group("text").strip()
            time_expr = mh1.group("time_expr").strip()
        elif mh2:
            message = mh2.group("text").strip()
            time_expr = mh2.group("time_expr").strip()
        elif mh3:
            message = mh3.group("text").strip()
            time_expr = mh3.group("time_expr").strip()
        elif mh_fb:
            message = mh_fb.group("text").strip()
        else:
            # Fall back to English parser
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

            m_sec = re.search(r"(\d+)\s*(second|sec|seconds|सेकंड|सेकण्ड)", time_expr, re.IGNORECASE)
            if m_sec:
                seconds = int(m_sec.group(1))

            m_min = re.search(r"(\d+)\s*(minute|min|minutes|मिनट|मिनिट)", time_expr, re.IGNORECASE)
            if m_min:
                minutes = int(m_min.group(1))

            m_hr = re.search(r"(\d+)\s*(hour|hr|hours|घंटे|घंटा|आर)", time_expr, re.IGNORECASE)
            if m_hr:
                hours = int(m_hr.group(1))

            is_relative = (
                time_expr.lower().startswith("in") or 
                time_expr.lower().startswith("after") or 
                "baad" in time_expr.lower() or 
                "mein" in time_expr.lower() or 
                "me" in time_expr.lower() or
                "बाद" in time_expr or 
                "में" in time_expr
            )
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

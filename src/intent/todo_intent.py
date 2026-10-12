"""
Morphix Desktop Companion Robot - Local Todo Intent Parser
==========================================================
File: src/intent/todo_intent.py
Description:
    Rule-based local parser that handles natural voice commands for To-Do operations:
    Add, Show, Delete single, and Clear all, utilizing regex.
    Preserves original casing of extracted task details.
"""

import re
from typing import Any, Dict


class TodoIntentParser:
    """
    Analyzes normalized voice commands to classify and parse To-Do action intents.
    """

    @staticmethod
    def parse(clean_text: str) -> Dict[str, Any]:
        """
        Extracts structured intent and item descriptions from voice command transcripts.
        Preserves original casing of the task name.
        Supports English, Hindi, and Hinglish.

        :param clean_text: Cleaned voice transcript string.
        :return: Dict containing parsed intent (ADD_TODO, SHOW_TODO, DELETE_TODO, CLEAR_TODO) and data.
        """
        if not clean_text:
            return {"intent": "UNKNOWN"}

        text = clean_text.strip()
        text_lower = text.lower()

        # Helper to strip trailing suffixes from task descriptions
        def strip_todo_suffixes(item_text: str) -> str:
            cleaned = re.sub(
                r"\s+(?:to|in|into|on|from)\s+(?:my\s+)?(?:todo|to-?do|tasks?)(?:\s+list)?$",
                "",
                item_text,
                flags=re.IGNORECASE
            )
            return cleaned.strip()

        # 1. CLEAR ALL
        # Pattern to trigger CLEAR_TODO intent
        clear_pattern = (
            r"^(?:delete|clear|remove|empty)\s+all\s+(?:my\s+)?todos?(?:\s+list)?$"
            r"|^(?:clear|empty)\s+(?:my\s+)?todo(?:\s+list)?$"
            r"|^saare\s+todo\s+(?:delete|hata|clear)(?:\s+(?:kar\s+)?do|\s+karo)?$"
        )
        if re.search(clear_pattern, text_lower):
            return {"intent": "CLEAR_TODO"}

        # 2. DELETE SINGLE
        # Match Hindi/Hinglish delete pattern first: <task> todo (se) delete/hata/remove (kar do)
        delete_pattern_hindi = r"^(?P<task>.+?)\s+(?:todo|list)\s+(?:se\s+)?(?:delete|hata|remove)(?:\s+(?:kar\s+)?do|\s*karo)?$"
        m_delete_hindi = re.match(delete_pattern_hindi, text_lower)
        if m_delete_hindi:
            start, end = m_delete_hindi.span("task")
            item = text[start:end].strip()
            return {
                "intent": "DELETE_TODO",
                "item": item
            }

        # Match English verbs like: delete, remove, complete, finish
        delete_pattern = r"^(?:delete|remove|complete|finish)\s+(.+)$"
        m_delete = re.match(delete_pattern, text_lower)
        if m_delete:
            start, end = m_delete.span(1)
            item = text[start:end].strip()
            item_cleaned = strip_todo_suffixes(item)
            if item_cleaned:
                return {
                    "intent": "DELETE_TODO",
                    "item": item_cleaned
                }

        # 3. ADD
        # Match Hindi/Hinglish add pattern first: todo mein <task> daal do/add karo/likh do
        add_pattern_hindi = r"^(?:todo|list)\s+(?:mein|me)\s+(?P<task>.+?)\s+(?:daal do|add karo|likh do|add kar do|karo)$"
        m_add_hindi = re.match(add_pattern_hindi, text_lower)
        if m_add_hindi:
            start, end = m_add_hindi.span("task")
            item = text[start:end].strip()
            return {
                "intent": "ADD_TODO",
                "item": item
            }

        # Match English verbs like: add, put
        add_pattern1 = r"^add\s+(?:task\s+)?(.+)$"
        add_pattern2 = r"^put\s+(.+)$"
        
        m_add1 = re.match(add_pattern1, text_lower)
        if m_add1:
            start, end = m_add1.span(1)
            item = text[start:end].strip()
            item_cleaned = strip_todo_suffixes(item)
            if item_cleaned:
                return {
                    "intent": "ADD_TODO",
                    "item": item_cleaned
                }
            
        m_add2 = re.match(add_pattern2, text_lower)
        if m_add2:
            start, end = m_add2.span(1)
            item = text[start:end].strip()
            item_cleaned = strip_todo_suffixes(item)
            if item_cleaned:
                return {
                    "intent": "ADD_TODO",
                    "item": item_cleaned
                }

        # 4. SHOW / READ
        # Match English show/list, and Hindi/Hinglish show (e.g. "mera todo dikhao")
        show_pattern = (
            r"^(?:show|read|tell|what\s+are|list)\s+(?:me\s+)?(?:my\s+)?todos?(?:\s+list)?$"
            r"|^show\s+me\s+my\s+(?:todos?|tasks?)(?:\s+list)?$"
            r"|^show\s+to-?do\s+list$"
            r"|^what(?:'s| is)?\s+on\s+my\s+(?:todos?|tasks?)(?:\s+list)?$"
            r"|^my\s+(?:todos?|tasks?)\s+list$"
            r"|^read\s+my\s+todo$"
            r"|^(?:mera|mere)?\s*todo\s+(?:dikhao|batao|kya\s+kya\s+hai|kya hai)$"
        )
        if re.search(show_pattern, text_lower):
            return {"intent": "SHOW_TODO"}

        return {"intent": "UNKNOWN"}

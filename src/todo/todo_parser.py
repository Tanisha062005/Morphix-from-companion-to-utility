"""
Morphix Desktop Companion Robot - Offline To-Do Command Parser
==============================================================
File: src/todo/todo_parser.py
Description:
    Parses cleaned transcripts and extracts structured task intents and items
    for the Offline To-Do Management system.
"""

import re
from typing import Any, Dict


class TodoParser:
    """
    Analyzes lowercase cleaned transcripts to identify and parse To-Do action details.
    """

    @staticmethod
    def parse(clean_text: str) -> Dict[str, Any]:
        """
        Extracts structured intent, item description, and old/new parameters
        from a cleaned transcript.

        :param clean_text: Cleaned lowercase voice transcript string.
        :return: Dict containing parsed intent (e.g. ADD_TODO, COMPLETE_TODO, etc.)
                 and corresponding data fields.
        """
        if not clean_text:
            return {"intent": "UNKNOWN"}

        text = clean_text.strip()

        # 1. EDIT_TODO (edit, rename, change, modify)
        # Match "edit/rename/change/modify <old_item> to <new_item>"
        m_edit = re.search(r"^(?:edit|rename|change|modify)\s+(.+?)\s+to\s+(.+?)$", text)
        if m_edit:
            return {
                "intent": "EDIT_TODO",
                "old_item": m_edit.group(1).strip(),
                "new_item": m_edit.group(2).strip()
            }

        # 2. SHOW_TODO (show, list, what are my todos/tasks)
        show_patterns = [
            r"^show\s+my\s+todos?$",
            r"^show\s+me\s+my\s+todos?$",
            r"^show\s+me\s+my\s+tasks$",
            r"^list\s+my\s+tasks$",
            r"^list\s+my\s+todos?$",
            r"^what\s+are\s+my\s+todos?$",
            r"^show\s+todos?$",
            r"^list\s+tasks$",
            r"^show\s+tasks$"
        ]
        for pattern in show_patterns:
            if re.search(pattern, text):
                return {"intent": "SHOW_TODO"}

        # 3. ADD_TODO (add, please add, can you add, could you add, remember, remember to, put ... in my todo, create a task)
        # Match "put <item> in my todo/todolist"
        m_put = re.search(r"^put\s+(.+?)\s+in\s+my\s+todo(?:list)?$", text)
        if m_put:
            return {"intent": "ADD_TODO", "item": m_put.group(1).strip()}

        # Match general ADD prefixes
        m_add = re.search(
            r"^(?:please\s+add|can\s+you\s+add|could\s+you\s+add|add|remember\s+to|remember|create\s+a\s+task)\s+(.+?)$",
            text
        )
        if m_add:
            item = m_add.group(1).strip()
            # Clean off trailing location strings if caught inside the capture group
            if item.endswith(" in my todo"):
                item = item[:-11].strip()
            elif item.endswith(" in my todo list"):
                item = item[:-16].strip()
            return {"intent": "ADD_TODO", "item": item}

        # 4. DELETE_TODO (delete, remove, remove my, delete my)
        m_delete = re.search(r"^(?:delete\s+my|remove\s+my|delete|remove)\s+(.+?)$", text)
        if m_delete:
            return {"intent": "DELETE_TODO", "item": m_delete.group(1).strip()}

        # 5. COMPLETE_TODO (complete, mark completed, mark done, tick, done, done with)
        # Match "mark <item> as completed/done" or "mark <item> completed/done"
        m_mark = re.search(r"^mark\s+(.+?)\s+(?:as\s+)?(?:completed|done)$", text)
        if m_mark:
            return {"intent": "COMPLETE_TODO", "item": m_mark.group(1).strip()}

        # Match "done with <item>" or "done <item>"
        m_done = re.search(r"^done\s+(?:with\s+)?(.+?)$", text)
        if m_done:
            return {"intent": "COMPLETE_TODO", "item": m_done.group(1).strip()}

        # Match "complete/tick <item>"
        m_complete = re.search(r"^(?:complete|tick)\s+(.+?)$", text)
        if m_complete:
            return {"intent": "COMPLETE_TODO", "item": m_complete.group(1).strip()}

        return {"intent": "UNKNOWN"}

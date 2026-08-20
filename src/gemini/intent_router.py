"""
Morphix Desktop Companion Robot - Intent Router
===============================================
File: src/gemini/intent_router.py
Description:
    Dispatches validated Gemini intents to the local managers
    (TodoManager, ReminderManager, or Chat Memory) to producespoken responses.
"""

import logging
from typing import Any, Dict
from src.todo.todo_manager import TodoManager
from src.reminder.reminder_manager import ReminderManager

logger = logging.getLogger("Morphix.IntentRouter")


class IntentRouter:
    """
    Decides how to execute the intent locally using Morphix managers.
    Contains no local business logic itself.
    """

    def __init__(
        self,
        todo_mgr: TodoManager,
        reminder_mgr: ReminderManager,
        gemini_engine: Any
    ) -> None:
        self.todo_mgr = todo_mgr
        self.reminder_mgr = reminder_mgr
        self.gemini_engine = gemini_engine

    def is_available(self) -> bool:
        """
        Returns True if the underlying Gemini engine client is available.
        """
        if hasattr(self.gemini_engine, "client"):
            return self.gemini_engine.client.is_available()
        return False

    def route(self, validated_intent: Dict[str, Any]) -> str:
        """
        Executes operations based on validated intent mapping.

        :param validated_intent: Validated structured dictionary.
        :return: spoken response message for Piper TTS.
        """
        intent = validated_intent.get("intent", "UNKNOWN")

        if intent == "ADD_TODO":
            task = validated_intent.get("task", "")
            try:
                self.todo_mgr.add_todo(task)
                return "Todo added."
            except ValueError:
                return "Task already exists."

        elif intent == "DELETE_TODO":
            task = validated_intent.get("task", "")
            deleted = self.todo_mgr.delete_todo(task)
            return "Todo deleted." if deleted else "Todo not found."

        elif intent == "EDIT_TODO":
            old = validated_intent.get("old_item", "")
            new = validated_intent.get("new_item", "")
            updated = self.todo_mgr.edit_todo(old, new)
            return "Todo updated." if updated else "Todo not found."

        elif intent == "COMPLETE_TODO":
            task = validated_intent.get("task", "")
            completed = self.todo_mgr.complete_todo(task)
            return "Todo completed." if completed else "Todo not found."

        elif intent == "SHOW_TODO":
            pending = self.todo_mgr.get_pending_todos()
            if not pending:
                return "Todo list is empty."
            task_list_str = ", ".join(f"{i+1}. {t}" for i, t in enumerate(pending))
            return f"You have {len(pending)} pending tasks: {task_list_str}."

        elif intent == "CLEAR_TODO":
            self.todo_mgr.clear_all_todos()
            return "All tasks have been cleared from your todo list."

        elif intent == "SET_REMINDER":
            msg = validated_intent.get("message", "")
            delay_seconds = int(validated_intent.get("delay_seconds", 0))
            self.reminder_mgr.add_reminder_by_delay(msg, delay_seconds)
            return "Reminder saved."

        elif intent == "SHOW_REMINDERS":
            pending = self.reminder_mgr.get_pending_reminders()
            if not pending:
                return "Reminder list is empty."
            rem_list_str = ", ".join(f"{r.get('text')}" for r in pending)
            return f"You have {len(pending)} pending reminders: {rem_list_str}."

        elif intent == "DELETE_REMINDER":
            target = validated_intent.get("target", "")
            deleted = self.reminder_mgr.delete_reminder(target)
            return "Reminder deleted." if deleted else "Reminder not found."

        elif intent == "CHAT":
            query = validated_intent.get("query", "")
            return self.gemini_engine.generate_chat_response(query)

        elif intent == "WEATHER":
            return "I cannot check the weather yet."
        elif intent == "CALENDAR":
            return "I cannot access the calendar yet."
        elif intent == "VISION":
            return "I cannot analyze images yet."
        elif intent == "CAMERA":
            return "I cannot control the camera yet."
        elif intent == "MUSIC":
            return "I cannot play music yet."
        elif intent == "ROBOT_CONTROL":
            return "I cannot control the robot movements yet."
        elif intent == "WEB_SEARCH":
            return "I cannot search the web yet."

        return "I'm not sure how to do that yet."

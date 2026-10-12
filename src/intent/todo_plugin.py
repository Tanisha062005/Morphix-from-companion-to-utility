"""
Morphix Desktop Companion Robot - To-Do List Intent Plugin & Service
======================================================================
File: src/intent/todo_plugin.py
Description:
    Provides NLU intent matching for To-Do operations (ADD, SHOW, DELETE, COMPLETE, CLEAR)
    and integrates with TodoManager and the OLED display overlay system (EyeAnimator).
"""

import logging
import re
import sys
import threading
from typing import Any, Dict, List, Optional

from src.display.eye_animator import get_active_display
from src.todo.todo_manager import TodoManager

# Setup logger for Todo Plugin
logger = logging.getLogger("Morphix.TodoPlugin")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [TodoPlugin]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class TodoService:
    """
    Manages To-Do business operations and triggers display overlay cards on the OLED screen.
    """
    _instance: Optional["TodoService"] = None
    _lock = threading.Lock()

    def __init__(self, todo_manager: Optional[TodoManager] = None) -> None:
        self.todo_manager = todo_manager or TodoManager()

    @classmethod
    def get_instance(cls) -> "TodoService":
        with cls._lock:
            if cls._instance is None:
                cls._instance = TodoService()
            return cls._instance

    def show_todo_list(self, is_hindi: bool = False, display: Optional[Any] = None) -> str:
        """
        Retrieves pending tasks, triggers on-screen Pygame overlay card,
        and returns formatted spoken response for TTS.
        """
        pending = self.todo_manager.get_pending_tasks()
        task_strings = [str(item.get("task", "")).strip() for item in pending if item.get("task")]

        # Trigger display overlay — timeout generous enough to survive TTS readback
        active_disp = display or get_active_display()
        if active_disp:
            try:
                if hasattr(active_disp, "set_overlay"):
                    active_disp.set_overlay("TODO_VIEW", tasks=task_strings, timeout=10.0)
                elif hasattr(active_disp, "show_todo_overlay"):
                    active_disp.show_todo_overlay(task_strings, timeout=10.0)
                logger.info("Sent %d pending tasks to display overlay.", len(task_strings))
            except Exception as ex:
                logger.warning("Failed to render todo overlay: %s", ex)

        if not task_strings:
            return "Aapki todo list khali hai." if is_hindi else "Your todo list is empty."

        task_list_fmt = ", ".join(f"{i+1}. {t}" for i, t in enumerate(task_strings))
        if is_hindi:
            return f"Aapke paas {len(task_strings)} pending tasks hain: {task_list_fmt}."
        else:
            return f"You have {len(task_strings)} pending tasks: {task_list_fmt}."

    def add_task(self, task_text: str, is_hindi: bool = False, display: Optional[Any] = None) -> str:
        """Adds a task to To-Do list."""
        if not task_text or not task_text.strip():
            return "Kripya task details batayein." if is_hindi else "Please specify task details."

        try:
            self.todo_manager.add_task(task_text.strip())
            # Show updated list on display
            self.show_todo_list(is_hindi=is_hindi, display=display)
            return f"Task added: {task_text.strip()}." if not is_hindi else f"Task add kar diya: {task_text.strip()}."
        except ValueError:
            return "Yeh task pehle se list mein hai." if is_hindi else "Task already exists in your list."
        except Exception as ex:
            logger.error("Error adding todo task: %s", ex)
            return "Task add karne mein error aaya." if is_hindi else "Failed to add task."

    def delete_task(self, task_text: str, is_hindi: bool = False) -> str:
        """Deletes a task from To-Do list."""
        deleted = self.todo_manager.delete_task(task_text)
        if deleted:
            return f"Task delete kar diya: {task_text}." if is_hindi else f"Deleted task: {task_text}."
        return "Task nahi mila." if is_hindi else "Task not found."

    def complete_task(self, task_text: str, is_hindi: bool = False) -> str:
        """Marks a task as completed."""
        completed = self.todo_manager.complete_task(task_text)
        if completed:
            return f"Task complete ho gaya: {task_text}." if is_hindi else f"Marked completed: {task_text}."
        return "Task nahi mila." if is_hindi else "Task not found."

    def clear_all(self, is_hindi: bool = False, display: Optional[Any] = None) -> str:
        """Clears all tasks and shows a brief on-screen confirmation overlay."""
        self.todo_manager.clear_all_tasks()
        # Show a quick empty-list card to confirm visually
        active_disp = display or get_active_display()
        if active_disp:
            try:
                if hasattr(active_disp, "set_overlay"):
                    active_disp.set_overlay("TODO_VIEW", tasks=[], timeout=4.0)
                elif hasattr(active_disp, "show_todo_overlay"):
                    active_disp.show_todo_overlay([], timeout=4.0)
            except Exception as ex:
                logger.warning("Failed to render clear-todo overlay: %s", ex)
        return "Saare tasks clear kar diye." if is_hindi else "Your todo list has been cleared."


class TodoIntentPlugin:
    """
    NLU plugin handling To-Do list operations:
    SHOW_TODO, ADD_TODO, DELETE_TODO, COMPLETE_TODO, CLEAR_TODO.
    """

    def __init__(self, service: Optional[TodoService] = None) -> None:
        self.service = service or TodoService.get_instance()

    def get_todos(self) -> List[str]:
        """Returns list of pending task descriptions."""
        pending = self.service.todo_manager.get_pending_tasks()
        return [str(item.get("task", "")).strip() for item in pending if item.get("task")]

    def handle(self, intent_result: Any, display: Optional[Any] = None) -> str:
        """
        Executes the matching To-Do operation.
        When SHOW_TODO triggers, explicitly calls eye_animator.set_overlay('TODO_VIEW', tasks=self.get_todos())
        BEFORE invoking TTS so the visual card appears immediately as speech starts.
        """
        active_disp = display or get_active_display()
        intent = getattr(intent_result, "intent", intent_result)
        params = getattr(intent_result, "parameters", {}) if hasattr(intent_result, "parameters") else {}
        is_hindi = params.get("is_hindi", False)

        from src.intent.intent_classifier import IntentType
        if intent in (IntentType.SHOW_TODO, "SHOW_TODO"):
            return self.service.show_todo_list(is_hindi=is_hindi, display=active_disp)

        elif intent in (IntentType.CLEAR_TODO, "CLEAR_TODO"):
            return self.service.clear_all(is_hindi=is_hindi, display=active_disp)

        elif intent in (IntentType.ADD_TODO, "ADD_TODO"):
            task_str = params.get("task", "")
            return self.service.add_task(task_str, is_hindi=is_hindi, display=active_disp)

        elif intent in (IntentType.DELETE_TODO, "DELETE_TODO"):
            task_str = params.get("task", "")
            return self.service.delete_task(task_str, is_hindi=is_hindi)

        elif intent in (IntentType.COMPLETE_TODO, "COMPLETE_TODO"):
            task_str = params.get("task", "")
            return self.service.complete_task(task_str, is_hindi=is_hindi)

        return "I'm not sure how to do that." if not is_hindi else "Mujhe samajh nahi aaya."

    def match(self, command: str) -> Optional[Any]:
        from src.intent.intent_classifier import IntentResult, IntentType

        if not command or not isinstance(command, str):
            return None

        text = command.strip()
        lowercased = text.lower()
        is_hindi = bool(re.search(r'\b(mera|mere|meri|dikha|dikhao|bata|batao|kya|hai|daal do|add karo|hata do|delete karo)\b', lowercased))

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
                return IntentResult(
                    intent=IntentType.CLEAR_TODO,
                    confidence=1.0,
                    parameters={"is_hindi": is_hindi},
                    raw_text=text
                )

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
                    confidence=0.95,
                    parameters={"task": task_str, "is_hindi": is_hindi},
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
                        confidence=0.90,
                        parameters={"task": task_str, "is_hindi": is_hindi},
                        raw_text=text
                    )

        # 4. Explicit ADD_TODO (with 'add ... to/in todo' or Hindi 'todo mein ... daal do')
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
                        confidence=0.95,
                        parameters={"task": task_str, "is_hindi": is_hindi},
                        raw_text=text
                    )

        # 5. SHOW_TODO / READ_TODO (Robust Hinglish & English matching)
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
        # Only treat as SHOW_TODO if not an explicit creation command
        if not re.match(r"^(?:add|insert|likh|daal)\b", lowercased):
            for pat in show_patterns:
                if re.search(pat, lowercased):
                    return IntentResult(
                        intent=IntentType.SHOW_TODO,
                        confidence=1.0,
                        parameters={"is_hindi": is_hindi},
                        raw_text=text
                    )

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
                        confidence=0.85,
                        parameters={"task": task_str, "is_hindi": is_hindi},
                        raw_text=text
                    )

        return None

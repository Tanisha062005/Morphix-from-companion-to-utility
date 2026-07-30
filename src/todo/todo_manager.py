"""
Morphix Desktop Companion Robot - Offline To-Do Manager
========================================================
File: src/todo/todo_manager.py
Description:
    Provides offline task management capabilities for Morphix.
    Supports task creation, retrieval, status updates (completion), deletion,
    and clearing stored tasks safely in a persistent local JSON file.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union
import uuid

# Setup logger for TodoManager module
logger = logging.getLogger("Morphix.Todo")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [Todo]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class TodoManager:
    """
    Manages offline To-Do operations and persistent JSON storage for Morphix.
    """

    def __init__(self, file_path: Optional[str] = None) -> None:
        """
        Initializes TodoManager and resolves storage file path.

        :param file_path: Custom path to JSON file (default: storage/todo.json).
        """
        self.project_root: Path = self._find_project_root()
        if file_path:
            p = Path(file_path)
            self.file_path: Path = p if p.is_absolute() else self.project_root / p
        else:
            self.file_path: Path = self.project_root / "storage" / "todo.json"

        # Ensure parent storage directory exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_file_exists()

    def _find_project_root(self) -> Path:
        """Locates project root directory using pathlib."""
        current_file = Path(__file__).resolve()
        for parent in current_file.parents:
            if (parent / "src").exists() or (parent / "config.py").exists():
                return parent
        return current_file.parents[2]

    def _ensure_file_exists(self) -> None:
        """Creates or resets JSON storage file safely if corrupted."""
        if not self.file_path.exists():
            self._write_to_file([])
            return

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    logger.warning("Corrupted todo.json format detected. Resetting to empty list.")
                    self._write_to_file([])
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Error reading todo.json (%s). Resetting storage.", str(e))
            self._write_to_file([])

    def _read_from_file(self) -> List[Dict[str, Any]]:
        """Reads list of task dictionaries from storage/todo.json."""
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def _write_to_file(self, data: List[Dict[str, Any]]) -> bool:
        """Writes task dictionaries to storage/todo.json."""
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            return True
        except OSError as e:
            logger.error("Failed to write to todo.json: %s", str(e))
            return False

    def add_task(self, task_text: str) -> Dict[str, Any]:
        """
        Creates and stores a new To-Do task with UUID and ISO8601 timestamp.

        :param task_text: Task description string.
        :return: Created task dictionary.
        """
        clean_text = task_text.strip() if task_text else "Untitled Task"
        task_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat(timespec="seconds")

        task_entry: Dict[str, Any] = {
            "id": task_id,
            "task": clean_text,
            "created_at": created_at,
            "completed": False
        }

        tasks = self._read_from_file()
        tasks.append(task_entry)
        self._write_to_file(tasks)

        logger.info("Added task: '%s' (ID: %s)", clean_text, task_id)
        return task_entry

    def get_pending_tasks(self) -> List[Dict[str, Any]]:
        """Returns list of uncompleted tasks."""
        tasks = self._read_from_file()
        return [t for t in tasks if not t.get("completed", False)]

    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Returns list of all tasks."""
        return self._read_from_file()

    def complete_task(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Marks a task as completed matching task string or UUID.

        :param query: Task description or ID query string.
        :return: Completed task dictionary if found, None otherwise.
        """
        if not query:
            return None

        clean_query = query.strip().lower()
        tasks = self._read_from_file()
        matched_task = None

        for t in tasks:
            t_id = str(t.get("id", "")).lower()
            t_text = str(t.get("task", "")).lower()

            if clean_query == t_id or clean_query in t_text or t_text in clean_query:
                t["completed"] = True
                matched_task = t
                break

        if matched_task:
            self._write_to_file(tasks)
            logger.info("Marked task completed: '%s'", matched_task.get("task"))
            return matched_task

        logger.info("No matching task found to complete for query: '%s'", query)
        return None

    def delete_task(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Deletes a task matching description string or UUID.

        :param query: Task description or ID query string.
        :return: Deleted task dictionary if found, None otherwise.
        """
        if not query:
            return None

        clean_query = query.strip().lower()
        tasks = self._read_from_file()
        deleted_task = None
        remaining_tasks = []

        for t in tasks:
            t_id = str(t.get("id", "")).lower()
            t_text = str(t.get("task", "")).lower()

            if not deleted_task and (clean_query == t_id or clean_query in t_text or t_text in clean_query):
                deleted_task = t
            else:
                remaining_tasks.append(t)

        if deleted_task:
            self._write_to_file(remaining_tasks)
            logger.info("Deleted task: '%s'", deleted_task.get("task"))
            return deleted_task

        logger.info("No matching task found to delete for query: '%s'", query)
        return None

    def clear_all_tasks(self) -> int:
        """
        Clears all tasks from storage.

        :return: Number of tasks cleared.
        """
        tasks = self._read_from_file()
        count = len(tasks)
        self._write_to_file([])
        logger.info("Cleared all %d tasks from todo list.", count)
        return count

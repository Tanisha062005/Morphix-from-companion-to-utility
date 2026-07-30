"""
Morphix Desktop Companion Robot - Offline To-Do Manager
========================================================
File: src/reminders/todo.py
Description:
    Provides offline task management capabilities for Morphix,
    allowing creation, deletion, status updating, and querying of To-Do items
    stored safely in a local JSON file.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid


class TodoManager:
    """
    Manages offline To-Do task operations and JSON persistence for Morphix.
    """

    def __init__(self, file_path: str = "todo.json") -> None:
        """
        Initializes the TodoManager and sets up the JSON storage file.

        :param file_path: Path to the JSON storage file (default: "todo.json").
        """
        self.file_path: Path = Path(file_path)

        # Ensure directory exists if a subfolder path is passed
        if self.file_path.parent:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)

        self._ensure_file_exists()

    def _ensure_file_exists(self) -> None:
        """
        Creates the JSON storage file if missing, or resets it safely if corrupted.
        """
        if not self.file_path.exists():
            self._write_to_file([])
            return

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    self._write_to_file([])
        except (json.JSONDecodeError, OSError):
            self._write_to_file([])

    def _read_from_file(self) -> List[Dict[str, Any]]:
        """
        Reads and returns the list of tasks from the JSON storage file.

        :return: List of task dictionaries.
        """
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                return []
        except (json.JSONDecodeError, OSError):
            return []

    def _write_to_file(self, data: List[Dict[str, Any]]) -> bool:
        """
        Writes the list of tasks to the JSON storage file.

        :param data: List of task dictionaries to write.
        :return: True if writing succeeded, False otherwise.
        """
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            return True
        except OSError:
            return False

    def add_task(
        self,
        title: str,
        description: str = "",
        category: str = "General",
        priority: str = "Medium",
        due_date: str = ""
    ) -> Dict[str, Any]:
        """
        Creates and stores a new To-Do task.

        :param title: Short title of the task.
        :param description: Detailed description of the task.
        :param category: Category/tag (default: "General").
        :param priority: Importance level ("Low", "Medium", "High").
        :param due_date: Due date in YYYY-MM-DD format (optional).
        :return: Dictionary representation of the created task.
        """
        now = datetime.now()

        task: Dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "title": title.strip(),
            "description": description.strip(),
            "category": category.strip() if category.strip() else "General",
            "priority": priority if priority in ["Low", "Medium", "High"] else "Medium",
            "status": "Pending",
            "due_date": due_date.strip(),
            "created_at": now.isoformat()
        }

        tasks = self._read_from_file()
        tasks.append(task)
        self._write_to_file(tasks)
        return task

    def remove_task(self, task_id: str) -> bool:
        """
        Removes a task by its unique ID.

        :param task_id: UUID string of the task to delete.
        :return: True if successfully removed, False if not found.
        """
        tasks = self._read_from_file()
        initial_count = len(tasks)
        tasks = [t for t in tasks if t.get("id") != task_id]

        if len(tasks) < initial_count:
            self._write_to_file(tasks)
            return True
        return False

    def update_task(self, task_id: str, **kwargs: Any) -> bool:
        """
        Updates specific attributes of an existing task.

        :param task_id: UUID string of the target task.
        :param kwargs: Key-value attributes to update (title, status, priority, etc.).
        :return: True if updated successfully, False if task not found.
        """
        tasks = self._read_from_file()
        for task in tasks:
            if task.get("id") == task_id:
                for key, value in kwargs.items():
                    if key in task and key != "id":
                        task[key] = value
                self._write_to_file(tasks)
                return True
        return False

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a single task by its unique ID.

        :param task_id: UUID string of the task.
        :return: Task dictionary if found, None otherwise.
        """
        tasks = self._read_from_file()
        for task in tasks:
            if task.get("id") == task_id:
                return task
        return None

    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """
        Retrieves all stored tasks.

        :return: List of all task dictionaries.
        """
        return self._read_from_file()

    def mark_completed(self, task_id: str) -> bool:
        """
        Marks a specific task status as "Completed".

        :param task_id: UUID string of the task.
        :return: True if updated successfully, False if not found.
        """
        return self.update_task(task_id, status="Completed")

    def mark_in_progress(self, task_id: str) -> bool:
        """
        Marks a specific task status as "In Progress".

        :param task_id: UUID string of the task.
        :return: True if updated successfully, False if not found.
        """
        return self.update_task(task_id, status="In Progress")

    def get_pending_tasks(self) -> List[Dict[str, Any]]:
        """
        Retrieves all tasks that are currently "Pending" or "In Progress".

        :return: List of non-completed task dictionaries.
        """
        tasks = self._read_from_file()
        return [t for t in tasks if t.get("status") in ["Pending", "In Progress"]]

    def clear_all(self) -> bool:
        """
        Deletes all stored tasks.

        :return: True if successfully cleared, False otherwise.
        """
        return self._write_to_file([])

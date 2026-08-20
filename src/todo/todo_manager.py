"""
Morphix Desktop Companion Robot - Offline To-Do Manager
========================================================
File: src/todo/todo_manager.py
Description:
    Provides offline task management capabilities for Morphix.
    Tracks tasks as TodoItem objects, manages serialization to JSON,
    and supports strict normalized exact query matching to prevent substring errors.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
from rapidfuzz import fuzz

from src.todo.todo_models import TodoItem

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
    Manages offline To-Do operations, internal TodoItem models,
    and persistent JSON storage for Morphix.
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
            self.file_path: Path = self.project_root / "data" / "todos.json"

        self.tasks: List[TodoItem] = []
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
        """Creates or resets JSON storage file safely if corrupted, and populates task models."""
        if not self.file_path.exists():
            self._write_to_file([])
            return

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    logger.warning("Corrupted todo.json format detected. Resetting to empty list.")
                    self._write_to_file([])
                else:
                    self._populate_tasks_from_data(data)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Error reading todo.json (%s). Resetting storage.", str(e))
            self._write_to_file([])

    def _populate_tasks_from_data(self, data: List[Dict[str, Any]]) -> None:
        """Populates task objects list from raw JSON dictionary data."""
        self.tasks = []
        for d in data:
            self.tasks.append(TodoItem(
                id=d.get("id", ""),
                task=d.get("task", ""),
                created_at=d.get("created_at", ""),
                completed=d.get("completed", False),
                completed_at=d.get("completed_at"),
                updated_at=d.get("updated_at")
            ))

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

    def _save(self) -> bool:
        """Serializes current list of TodoItem objects and writes to file."""
        return self._write_to_file([task.to_dict() for task in self.tasks])

    def _find_matching_task(self, query: str) -> Optional[TodoItem]:
        """
        Locates task matching description query safely using normalized exact matching.
        Avoids unsafe substring matching.

        Priority order:
          1. exact normalized match (whitespace stripped, lowercase)
          2. case-insensitive match
          3. whitespace normalized match
        """
        if not query:
            return None

        # Whitespace and case-insensitive normalization
        norm_query = " ".join(query.strip().lower().split())

        # Priority 1: Match pending tasks first
        for t in self.tasks:
            if not t.completed:
                norm_task = " ".join(t.task.strip().lower().split())
                if norm_task == norm_query:
                    return t

        # Priority 2: Match completed tasks if not found in pending
        for t in self.tasks:
            norm_task = " ".join(t.task.strip().lower().split())
            if norm_task == norm_query:
                return t

        return None

    def add_task(self, task_text: str) -> Dict[str, Any]:
        """
        Creates and stores a new To-Do task.
        Prevents adding case-insensitive duplicate pending tasks.

        :param task_text: Task description string.
        :return: Created task dictionary.
        :raises ValueError: If duplicate pending task description exists.
        """
        clean_text = task_text.strip() if task_text else "Untitled Task"
        norm_text = " ".join(clean_text.lower().split())

        # Check duplicate pending tasks
        for t in self.tasks:
            if not t.completed and " ".join(t.task.strip().lower().split()) == norm_text:
                logger.warning("Attempted to add duplicate task: '%s'", clean_text)
                raise ValueError("Task already exists")

        import uuid
        task_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat(timespec="seconds")

        new_item = TodoItem(
            id=task_id,
            task=clean_text,
            created_at=created_at,
            completed=False,
            completed_at=None,
            updated_at=None
        )

        self.tasks.append(new_item)
        self._save()

        logger.info("Added task: '%s' (ID: %s)", clean_text, task_id)
        return new_item.to_dict()

    def get_pending_tasks(self) -> List[Dict[str, Any]]:
        """Returns list of uncompleted tasks as dictionaries."""
        return [t.to_dict() for t in self.tasks if not t.completed]

    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Returns list of all tasks as dictionaries."""
        return [t.to_dict() for t in self.tasks]

    def complete_task(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Marks a task as completed matching task string query safely.

        :param query: Task description or ID query string.
        :return: Completed task dictionary if found, None otherwise.
        """
        # Support matching by direct UUID first
        matched_task = None
        if query:
            clean_query = query.strip().lower()
            for t in self.tasks:
                if t.id.lower() == clean_query:
                    matched_task = t
                    break

        if not matched_task:
            matched_task = self._find_matching_task(query)

        if matched_task:
            matched_task.completed = True
            matched_task.completed_at = datetime.now().isoformat(timespec="seconds")
            self._save()
            logger.info("Marked task completed: '%s'", matched_task.task)
            return matched_task.to_dict()

        logger.info("No matching task found to complete for query: '%s'", query)
        return None

    def edit_task(self, query: str, new_task_text: str) -> Optional[Dict[str, Any]]:
        """
        Edits the text of an existing task.
        Automatically updates the 'updated_at' metadata field.

        :param query: Task description or ID query string.
        :param new_task_text: New description.
        :return: Updated task dictionary if found, None otherwise.
        """
        # Support matching by direct UUID first
        matched_task = None
        if query:
            clean_query = query.strip().lower()
            for t in self.tasks:
                if t.id.lower() == clean_query:
                    matched_task = t
                    break

        if not matched_task:
            matched_task = self._find_matching_task(query)

        if matched_task and new_task_text:
            matched_task.task = new_task_text.strip()
            matched_task.updated_at = datetime.now().isoformat(timespec="seconds")
            self._save()
            logger.info("Edited task (ID: %s) to: '%s'", matched_task.id, matched_task.task)
            return matched_task.to_dict()

        logger.info("No matching task found to edit for query: '%s'", query)
        return None

    def delete_task(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Deletes a task matching description string query safely.

        :param query: Task description or ID query string.
        :return: Deleted task dictionary if found, None otherwise.
        """
        # Support matching by direct UUID first
        matched_task = None
        if query:
            clean_query = query.strip().lower()
            for t in self.tasks:
                if t.id.lower() == clean_query:
                    matched_task = t
                    break

        if not matched_task:
            matched_task = self._find_matching_task(query)

        if matched_task:
            self.tasks.remove(matched_task)
            self._save()
            logger.info("Deleted task: '%s'", matched_task.task)
            return matched_task.to_dict()

        logger.info("No matching task found to delete for query: '%s'", query)
        return None

    def clear_all_tasks(self) -> int:
        """
        Clears all tasks from storage.

        :return: Number of tasks cleared.
        """
        count = len(self.tasks)
        self.tasks = []
        self._save()
        logger.info("Cleared all %d tasks from todo list.", count)
        return count

    # Core Methods to support Part 3 architecture
    def add_todo(self, task_text: str) -> str:
        """
        Creates and stores a new To-Do task.
        Prevents adding case-insensitive duplicate pending tasks.

        :param task_text: Task description string.
        :return: Confirmation string.
        :raises ValueError: If duplicate pending task description exists.
        """
        res = self.add_task(task_text)
        return f"Added '{res['task']}' to your todo list."

    def get_pending_todos(self) -> List[str]:
        """
        Returns list of pending task strings.

        :return: List of uncompleted task strings.
        """
        return [t.task for t in self.tasks if not t.completed]

    def delete_todo(self, query: str) -> bool:
        """
        Deletes a task matching description string query safely using substring/fuzzy matching.

        :param query: Task description or ID query string.
        :return: True if found and deleted, False otherwise.
        """
        if not query:
            return False

        clean_query = query.strip().lower()
        matched_task = None

        # Helper for word-level substring matching to prevent partial-word matches
        def is_word_substring(q: str, t: str) -> bool:
            q_words = q.split()
            t_words = t.split()
            if not q_words or not t_words:
                return False
            n, m = len(q_words), len(t_words)
            for idx in range(m - n + 1):
                if t_words[idx:idx+n] == q_words:
                    return True
            return False

        # 1. Match by direct UUID first (case-insensitive)
        for t in self.tasks:
            if t.id.lower() == clean_query:
                matched_task = t
                break

        # 2. Exact case-insensitive normalized match (pending first)
        if not matched_task:
            norm_query = " ".join(clean_query.split())
            for t in self.tasks:
                if not t.completed:
                    norm_task = " ".join(t.task.strip().lower().split())
                    if norm_task == norm_query:
                        matched_task = t
                        break

        # 3. Substring match (pending first, using word-level boundaries)
        if not matched_task:
            for t in self.tasks:
                if not t.completed:
                    norm_task = " ".join(t.task.strip().lower().split())
                    if is_word_substring(norm_query, norm_task):
                        matched_task = t
                        break

        # 4. Fuzzy match (pending first) using rapidfuzz
        if not matched_task:
            best_match = None
            best_score = 0.0
            for t in self.tasks:
                if not t.completed:
                    norm_task = " ".join(t.task.strip().lower().split())
                    score = fuzz.ratio(norm_query, norm_task)
                    if score > best_score:
                        best_score = score
                        best_match = t
            if best_score >= 80.0:
                matched_task = best_match

        # 5. Exact match on completed tasks (fallback)
        if not matched_task:
            for t in self.tasks:
                if t.completed:
                    norm_task = " ".join(t.task.strip().lower().split())
                    if norm_task == norm_query:
                        matched_task = t
                        break

        # 6. Substring match on completed tasks (fallback, using word-level boundaries)
        if not matched_task:
            for t in self.tasks:
                if t.completed:
                    norm_task = " ".join(t.task.strip().lower().split())
                    if is_word_substring(norm_query, norm_task):
                        matched_task = t
                        break

        # 7. Fuzzy match on completed tasks (fallback)
        if not matched_task:
            best_match = None
            best_score = 0.0
            for t in self.tasks:
                if t.completed:
                    norm_task = " ".join(t.task.strip().lower().split())
                    score = fuzz.ratio(norm_query, norm_task)
                    if score > best_score:
                        best_score = score
                        best_match = t
            if best_score >= 80.0:
                matched_task = best_match

        if matched_task:
            self.tasks.remove(matched_task)
            self._save()
            logger.info("Deleted task: '%s'", matched_task.task)
            return True

        logger.info("No matching task found to delete for query: '%s'", query)
        return False

    def clear_all_todos(self) -> int:
        """
        Removes all pending tasks. Returns count of cleared items.

        :return: Count of cleared tasks.
        """
        initial_count = len(self.tasks)
        self.tasks = [t for t in self.tasks if t.completed]
        cleared = initial_count - len(self.tasks)
        self._save()
        logger.info("Cleared all %d pending tasks.", cleared)
        return cleared

    def edit_todo(self, query: str, new_task_text: str) -> Optional[Dict[str, Any]]:
        """Alias for edit_task."""
        return self.edit_task(query, new_task_text)

    def complete_todo(self, query: str) -> Optional[Dict[str, Any]]:
        """Alias for complete_task."""
        return self.complete_task(query)

    def show_todos(self) -> List[Dict[str, Any]]:
        """Returns pending tasks list."""
        return self.get_pending_tasks()

    def load_json(self) -> List[Dict[str, Any]]:
        """Loads and returns all todos from JSON storage."""
        data = self._read_from_file()
        self._populate_tasks_from_data(data)
        return [t.to_dict() for t in self.tasks]

    def save_json(self, data: List[Dict[str, Any]]) -> bool:
        """Saves list of dictionaries to JSON storage and populates models."""
        res = self._write_to_file(data)
        if res:
            self._populate_tasks_from_data(data)
        return res

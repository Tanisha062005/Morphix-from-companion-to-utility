"""
Morphix Desktop Companion Robot - Reminder Manager
===================================================
File: src/reminders/reminder.py
Description:
    Provides an offline reminder management system for Morphix,
    allowing creation, retrieval, updates, and completion tracking of reminders
    stored safely in a local JSON file.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid


class ReminderManager:
    """
    Manages offline reminder creation, deletion, completion tracking,
    and persistent JSON storage for Morphix.
    """

    def __init__(self, file_path: str = "reminders.json") -> None:
        """
        Initializes the ReminderManager and prepares the JSON storage file.

        :param file_path: Path to the JSON storage file (default: "reminders.json").
        """
        self.file_path: Path = Path(file_path)

        # Ensure parent folder exists if a nested path is specified
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
        Reads and returns the list of reminders from the JSON file.

        :return: List of reminder dictionaries.
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
        Writes the list of reminders to the JSON storage file.

        :param data: List of reminder dictionaries to save.
        :return: True if writing succeeded, False otherwise.
        """
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            return True
        except OSError:
            return False

    def add_reminder(
        self,
        title: str,
        description: str = "",
        date: str = "",
        time: str = "",
        repeat: str = "None",
        priority: str = "Medium"
    ) -> Dict[str, Any]:
        """
        Creates and stores a new reminder.

        :param title: Short title of the reminder.
        :param description: Detailed description of the task.
        :param date: Date in YYYY-MM-DD format (defaults to today if empty).
        :param time: Time in HH:MM format (defaults to current time if empty).
        :param repeat: Recurrence rule ("None", "Daily", "Weekly", "Monthly").
        :param priority: Importance level ("Low", "Medium", "High").
        :return: Dictionary representation of the created reminder.
        """
        now = datetime.now()

        # Set default date and time if omitted
        target_date = date.strip() if date.strip() else now.strftime("%Y-%m-%d")
        target_time = time.strip() if time.strip() else now.strftime("%H:%M")

        reminder: Dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "title": title.strip(),
            "description": description.strip(),
            "date": target_date,
            "time": target_time,
            "repeat": repeat if repeat in ["None", "Daily", "Weekly", "Monthly"] else "None",
            "priority": priority if priority in ["Low", "Medium", "High"] else "Medium",
            "completed": False,
            "created_at": now.isoformat()
        }

        reminders = self._read_from_file()
        reminders.append(reminder)
        self._write_to_file(reminders)
        return reminder

    def remove_reminder(self, reminder_id: str) -> bool:
        """
        Removes a reminder by its unique ID.

        :param reminder_id: UUID string of the reminder to delete.
        :return: True if successfully removed, False if not found.
        """
        reminders = self._read_from_file()
        initial_length = len(reminders)
        reminders = [r for r in reminders if r.get("id") != reminder_id]

        if len(reminders) < initial_length:
            self._write_to_file(reminders)
            return True
        return False

    def update_reminder(self, reminder_id: str, **kwargs: Any) -> bool:
        """
        Updates fields of an existing reminder.

        :param reminder_id: UUID string of the target reminder.
        :param kwargs: Keyword arguments of fields to update (title, date, priority, etc.).
        :return: True if updated successfully, False if reminder not found.
        """
        reminders = self._read_from_file()
        for reminder in reminders:
            if reminder.get("id") == reminder_id:
                for key, value in kwargs.items():
                    if key in reminder and key != "id":
                        reminder[key] = value
                self._write_to_file(reminders)
                return True
        return False

    def get_reminder(self, reminder_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a single reminder by its ID.

        :param reminder_id: UUID string of the reminder.
        :return: Reminder dictionary if found, None otherwise.
        """
        reminders = self._read_from_file()
        for reminder in reminders:
            if reminder.get("id") == reminder_id:
                return reminder
        return None

    def get_all_reminders(self) -> List[Dict[str, Any]]:
        """
        Retrieves all stored reminders.

        :return: List of all reminder dictionaries.
        """
        return self._read_from_file()

    def mark_completed(self, reminder_id: str) -> bool:
        """
        Marks a specific reminder as completed.

        :param reminder_id: UUID string of the reminder.
        :return: True if updated, False if reminder not found.
        """
        return self.update_reminder(reminder_id, completed=True)

    def get_today_reminders(self) -> List[Dict[str, Any]]:
        """
        Retrieves all reminders scheduled for today's date.

        :return: List of reminders matching today's date.
        """
        today_str = datetime.now().strftime("%Y-%m-%d")
        reminders = self._read_from_file()
        return [r for r in reminders if r.get("date") == today_str]

    def clear_all(self) -> bool:
        """
        Deletes all stored reminders.

        :return: True if successfully cleared, False otherwise.
        """
        return self._write_to_file([])

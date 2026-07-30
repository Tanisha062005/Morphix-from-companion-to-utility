"""
Morphix Desktop Companion Robot - Offline Reminder Manager
===========================================================
File: src/reminder/reminder_manager.py
Description:
    Provides an offline reminder management system for Morphix.
    Converts absolute ("5 PM", "tomorrow at 9 AM") and relative ("in 10 minutes",
    "after 1 hour 30 minutes") time expressions into absolute ISO8601 timestamps before
    saving into persistent local JSON storage (storage/reminders.json).
"""

from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Union
import uuid

# Setup logger for ReminderManager module
logger = logging.getLogger("Morphix.Reminder")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [Reminder]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def parse_time_expression(time_expr: str, now: Optional[datetime] = None) -> datetime:
    """
    Parses absolute or relative time expressions into an absolute datetime object.

    :param time_expr: Time expression string (e.g., '5 PM', 'tomorrow at 9 AM', 'in 10 minutes').
    :param now: Base reference datetime object (defaults to datetime.now()).
    :return: Computed absolute datetime object.
    """
    if now is None:
        now = datetime.now()

    expr = time_expr.lower().strip() if time_expr else ""
    if not expr:
        return now + timedelta(minutes=10)

    # 1. Parse relative duration expressions (e.g., 'in 10 minutes', 'after 1 hour 30 minutes')
    hours = 0
    minutes = 0

    m_hrs = re.search(r"(\d+)\s*(hour|hr|hours)", expr)
    if m_hrs:
        hours = int(m_hrs.group(1))

    m_mins = re.search(r"(\d+)\s*(minute|min|minutes)", expr)
    if m_mins:
        minutes = int(m_mins.group(1))

    if hours > 0 or minutes > 0:
        return now + timedelta(hours=hours, minutes=minutes)

    # 2. Parse absolute time expressions (e.g., '5 PM', 'today at 8 PM', 'tomorrow at 9 AM')
    is_tomorrow = "tomorrow" in expr

    m_time = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", expr)
    if m_time:
        hr = int(m_time.group(1))
        mn = int(m_time.group(2)) if m_time.group(2) else 0
        ampm = m_time.group(3)

        if ampm:
            if ampm == "pm" and hr < 12:
                hr += 12
            elif ampm == "am" and hr == 12:
                hr = 0

        target_date = now.date()
        if is_tomorrow:
            target_date += timedelta(days=1)

        target_dt = datetime.combine(target_date, datetime.min.time().replace(hour=hr, minute=mn))
        if target_dt < now and not is_tomorrow:
            target_dt += timedelta(days=1)

        return target_dt

    return now + timedelta(minutes=10)


class ReminderManager:
    """
    Manages offline reminder operations and persistent JSON storage for Morphix.
    """

    def __init__(self, file_path: Optional[str] = None) -> None:
        """
        Initializes ReminderManager and sets up JSON storage.

        :param file_path: Custom path to JSON storage file (default: storage/reminders.json).
        """
        self.project_root: Path = self._find_project_root()
        if file_path:
            p = Path(file_path)
            self.file_path: Path = p if p.is_absolute() else self.project_root / p
        else:
            self.file_path: Path = self.project_root / "storage" / "reminders.json"

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
                    logger.warning("Corrupted reminders.json format detected. Resetting to empty list.")
                    self._write_to_file([])
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Error reading reminders.json (%s). Resetting storage.", str(e))
            self._write_to_file([])

    def _read_from_file(self) -> List[Dict[str, Any]]:
        """Reads list of reminder dictionaries from storage/reminders.json."""
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def _write_to_file(self, data: List[Dict[str, Any]]) -> bool:
        """Writes reminder dictionaries to storage/reminders.json."""
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            return True
        except OSError as e:
            logger.error("Failed to write to reminders.json: %s", str(e))
            return False

    def add_reminder(self, reminder_text: str, time_expr: str) -> Dict[str, Any]:
        """
        Creates and stores a new reminder with UUID, ISO8601 trigger time, and created timestamp.

        :param reminder_text: Text description of the reminder.
        :param time_expr: Absolute or relative time expression string.
        :return: Created reminder dictionary.
        """
        clean_text = reminder_text.strip() if reminder_text else "Reminder"
        trigger_dt = parse_time_expression(time_expr)
        reminder_id = str(uuid.uuid4())
        now_str = datetime.now().isoformat(timespec="seconds")
        trigger_str = trigger_dt.isoformat(timespec="seconds")

        reminder_entry: Dict[str, Any] = {
            "id": reminder_id,
            "text": clean_text,
            "trigger_time": trigger_str,
            "created_at": now_str,
            "completed": False,
            "repeat": "none"
        }

        reminders = self._read_from_file()
        reminders.append(reminder_entry)
        self._write_to_file(reminders)

        logger.info("Saved reminder: '%s' for trigger_time: %s (ID: %s)", clean_text, trigger_str, reminder_id)
        return reminder_entry

    def get_due_reminders(self, reference_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Returns list of uncompleted reminders whose trigger_time <= reference_time.

        :param reference_time: Datetime reference object (defaults to datetime.now()).
        :return: List of due reminder dictionaries.
        """
        if reference_time is None:
            reference_time = datetime.now()

        reminders = self._read_from_file()
        due_reminders = []

        for rem in reminders:
            if rem.get("completed", False):
                continue

            trig_str = rem.get("trigger_time", "")
            try:
                trig_dt = datetime.fromisoformat(trig_str)
                if trig_dt <= reference_time:
                    due_reminders.append(rem)
            except Exception as e:
                logger.error("Invalid trigger_time in reminder %s: %s", rem.get("id"), str(e))

        return due_reminders

    def mark_completed(self, reminder_id: str) -> bool:
        """
        Marks a reminder as completed by ID.

        :param reminder_id: UUID string of the reminder.
        :return: True if updated successfully, False otherwise.
        """
        reminders = self._read_from_file()
        updated = False

        for rem in reminders:
            if str(rem.get("id")) == str(reminder_id):
                rem["completed"] = True
                updated = True
                break

        if updated:
            self._write_to_file(reminders)
            logger.info("Marked reminder ID %s as completed.", reminder_id)

        return updated

    def get_all_reminders(self) -> List[Dict[str, Any]]:
        """Returns list of all reminders."""
        return self._read_from_file()

    def get_pending_reminders(self) -> List[Dict[str, Any]]:
        """Returns list of pending (uncompleted) reminders."""
        reminders = self._read_from_file()
        return [r for r in reminders if not r.get("completed", False)]

    def delete_reminder(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Deletes a reminder matching text description or UUID.

        :param query: Search query or UUID string.
        :return: Deleted reminder dictionary if found, None otherwise.
        """
        if not query:
            return None

        clean_query = query.strip().lower()
        reminders = self._read_from_file()
        deleted_rem = None
        remaining = []

        for r in reminders:
            r_id = str(r.get("id", "")).lower()
            r_text = str(r.get("text", "")).lower()

            if not deleted_rem and (clean_query == r_id or clean_query in r_text or r_text in clean_query):
                deleted_rem = r
            else:
                remaining.append(r)

        if deleted_rem:
            self._write_to_file(remaining)
            logger.info("Deleted reminder: '%s'", deleted_rem.get("text"))
            return deleted_rem

        return None

"""
Morphix Desktop Companion Robot - Offline Timezone-Aware Reminder Manager
========================================================================
File: src/reminder/reminder_manager.py
Description:
    Provides a robust timezone-aware reminder management system for Morphix.
    Converts absolute and relative time expressions into timezone-aware datetimes
    and persists them into JSON storage (storage/reminders.json).
"""

from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional
import uuid
from zoneinfo import ZoneInfo

# Setup logger for ReminderManager module
logger = logging.getLogger("Morphix.Reminder")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [Reminder]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def get_local_timezone() -> ZoneInfo:
    """
    Returns the operating system's local timezone.
    Works across Windows, macOS, and Linux without hardcoded values.
    """
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\TimeZoneInformation") as key:
                win_tz_name, _ = winreg.QueryValueEx(key, "TimeZoneKeyName")
                mapping = {
                    "India Standard Time": "Asia/Kolkata",
                    "Eastern Standard Time": "America/New_York",
                    "W. Europe Standard Time": "Europe/Berlin",
                    "GMT Standard Time": "Europe/London",
                    "Pacific Standard Time": "America/Los_Angeles",
                    "Central Standard Time": "America/Chicago",
                    "Mountain Standard Time": "America/Denver",
                }
                if win_tz_name in mapping:
                    return ZoneInfo(mapping[win_tz_name])
                return ZoneInfo(win_tz_name)
        except Exception:
            pass

    if hasattr(time, "tzname") and time.tzname:
        tz_str = time.tzname[0]
        mapping = {
            "IST": "Asia/Kolkata",
            "EST": "America/New_York",
            "EDT": "America/New_York",
            "CST": "America/Chicago",
            "PST": "America/Los_Angeles",
            "PDT": "America/Los_Angeles",
            "GMT": "Europe/London",
            "BST": "Europe/London",
            "UTC": "UTC",
        }
        if tz_str in mapping:
            return ZoneInfo(mapping[tz_str])

    try:
        tz = datetime.now().astimezone().tzinfo
        if tz is not None:
            return ZoneInfo(tz.tzname(None))
    except Exception:
        pass

    return ZoneInfo("UTC")


def parse_time_expression(time_expr: str, tz: Optional[ZoneInfo] = None, now: Optional[datetime] = None) -> datetime:
    """
    Parses absolute or relative time expressions into a datetime object.

    :param time_expr: Time expression string (e.g., '5 PM', 'tomorrow at 9 AM', 'in 10 minutes').
    :param tz: Local system ZoneInfo. If None, defaults to system timezone.
    :param now: Base reference datetime object.
    :return: Computed datetime object (timezone-aware if tz is provided or now is timezone-aware).
    """
    is_aware = False
    if tz is not None:
        is_aware = True
    else:
        tz = get_local_timezone()
        if now is not None and now.tzinfo is not None:
            is_aware = True

    if now is None:
        now = datetime.now(tz)
        is_aware = True
    else:
        if now.tzinfo is None:
            now = now.replace(tzinfo=tz)
        else:
            now = now.astimezone(tz)

    expr = time_expr.lower().strip() if time_expr else ""
    if not expr:
        res_dt = now + timedelta(minutes=10)
        return res_dt if is_aware else res_dt.replace(tzinfo=None)

    # 1. Parse relative duration expressions (e.g., 'in 10 minutes', 'after 1 hour 30 minutes')
    seconds = 0
    minutes = 0
    hours = 0

    m_secs = re.search(r"(\d+)\s*(second|sec|seconds)", expr)
    if m_secs:
        seconds = int(m_secs.group(1))

    m_mins = re.search(r"(\d+)\s*(minute|min|minutes)", expr)
    if m_mins:
        minutes = int(m_mins.group(1))

    m_hrs = re.search(r"(\d+)\s*(hour|hr|hours)", expr)
    if m_hrs:
        hours = int(m_hrs.group(1))

    if seconds > 0 or minutes > 0 or hours > 0:
        res_dt = now + timedelta(hours=hours, minutes=minutes, seconds=seconds)
        return res_dt if is_aware else res_dt.replace(tzinfo=None)

    # 2. Parse absolute time expressions (e.g., 'tomorrow', 'tomorrow at 5 pm', 'today 8 pm', 'next monday')
    is_tomorrow = "tomorrow" in expr
    is_next = "next" in expr

    days_ahead = 0
    weekday_map = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6
    }
    for day_name, day_idx in weekday_map.items():
        if day_name in expr:
            current_weekday = now.weekday()
            days_ahead = (day_idx - current_weekday) % 7
            if days_ahead == 0 and is_next:
                days_ahead = 7
            elif is_next:
                days_ahead = (day_idx - current_weekday) % 7
                if days_ahead <= 0:
                    days_ahead += 7
            elif days_ahead == 0:
                # If they say "monday" and it's already monday, default to upcoming monday (7 days)
                days_ahead = 7
            break

    target_date = now.date()
    if is_tomorrow:
        target_date += timedelta(days=1)
    elif days_ahead > 0:
        target_date += timedelta(days=days_ahead)

    hr = 9  # default hour
    mn = 0

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

    target_dt = datetime.combine(target_date, datetime.min.time().replace(hour=hr, minute=mn), tzinfo=tz)
    if target_dt < now and not is_tomorrow and days_ahead == 0:
        target_dt += timedelta(days=1)

    return target_dt if is_aware else target_dt.replace(tzinfo=None)


class ReminderManager:
    """
    Manages offline reminder operations, persistence, and timezone coordination.
    """

    def __init__(self, file_path: Optional[str] = None, tz: Optional[ZoneInfo] = None) -> None:
        """
        Initializes ReminderManager and registers timezone.

        :param file_path: Custom path to JSON storage file.
        :param tz: Custom ZoneInfo object.
        """
        self.project_root: Path = self._find_project_root()
        if file_path:
            p = Path(file_path)
            self.file_path: Path = p if p.is_absolute() else self.project_root / p
        else:
            self.file_path: Path = self.project_root / "storage" / "reminders.json"

        self.tz: ZoneInfo = tz or get_local_timezone()
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_file_exists()

    def _find_project_root(self) -> Path:
        """Locates project root directory."""
        current_file = Path(__file__).resolve()
        for parent in current_file.parents:
            if (parent / "src").exists() or (parent / "config.py").exists():
                return parent
        return current_file.parents[2]

    def _ensure_file_exists(self) -> None:
        """safely instantiates reminders file."""
        if not self.file_path.exists():
            self._write_to_file([])
            return

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    logger.warning("Corrupted reminders JSON format. Resetting.")
                    self._write_to_file([])
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Error reading reminders JSON (%s). Resetting storage.", str(e))
            self._write_to_file([])

    def _read_from_file(self) -> List[Dict[str, Any]]:
        """Reads reminders from storage."""
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def _write_to_file(self, data: List[Dict[str, Any]]) -> bool:
        """Writes reminders to storage."""
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            return True
        except OSError as e:
            logger.error("Failed to write to reminders JSON: %s", str(e))
            return False

    def add_reminder(self, message: str, time_expr: str) -> Dict[str, Any]:
        """
        Creates and stores a new reminder.

        :param message: Text content of the reminder.
        :param time_expr: timing expression.
        :return: Created reminder dictionary.
        """
        clean_text = message.strip() if message else "Reminder"
        trigger_dt = parse_time_expression(time_expr, self.tz)
        reminder_id = str(uuid.uuid4())

        now_str = datetime.now(self.tz).isoformat()
        trigger_str = trigger_dt.isoformat()

        # Backward compatibility with older tests and properties
        reminder_entry: Dict[str, Any] = {
            "id": reminder_id,
            "message": clean_text,
            "text": clean_text,
            "created_at": now_str,
            "trigger_time": trigger_str,
            "timezone": str(self.tz),
            "status": "pending",
            "completed": False,
            "repeat": "none"
        }

        reminders = self._read_from_file()
        reminders.append(reminder_entry)
        self._write_to_file(reminders)

        logger.info("Saved reminder: '%s' for trigger_time: %s (ID: %s)", clean_text, trigger_str, reminder_id)
        return reminder_entry

    def add_reminder_by_delay(self, message: str, delay_seconds: int) -> Dict[str, Any]:
        """
        Creates and stores a new reminder triggerable after delay_seconds.
        """
        clean_text = message.strip() if message else "Reminder"
        trigger_dt = datetime.now(self.tz) + timedelta(seconds=delay_seconds)
        reminder_id = str(uuid.uuid4())

        now_str = datetime.now(self.tz).isoformat()
        trigger_str = trigger_dt.isoformat()

        reminder_entry: Dict[str, Any] = {
            "id": reminder_id,
            "message": clean_text,
            "text": clean_text,
            "created_at": now_str,
            "trigger_time": trigger_str,
            "timezone": str(self.tz),
            "status": "pending",
            "completed": False,
            "repeat": "none"
        }

        reminders = self._read_from_file()
        reminders.append(reminder_entry)
        self._write_to_file(reminders)

        logger.info("Saved reminder: '%s' with delay: %s seconds for trigger_time: %s (ID: %s)", clean_text, delay_seconds, trigger_str, reminder_id)
        return reminder_entry

    def get_due_reminders(self, reference_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Returns list of pending reminders whose trigger_time <= reference_time.
        """
        if reference_time is None:
            reference_time = datetime.now(self.tz)

        reminders = self._read_from_file()
        due_reminders = []

        for rem in reminders:
            if rem.get("status", "pending") != "pending" and not rem.get("completed", False) == False:
                continue

            trig_str = rem.get("trigger_time", "")
            try:
                trig_dt = datetime.fromisoformat(trig_str)
                # Align timezones (naive vs aware datetimes)
                if trig_dt.tzinfo is None and reference_time.tzinfo is not None:
                    trig_dt = trig_dt.replace(tzinfo=reference_time.tzinfo)
                elif trig_dt.tzinfo is not None and reference_time.tzinfo is None:
                    reference_time = reference_time.replace(tzinfo=trig_dt.tzinfo)

                if trig_dt <= reference_time:
                    due_reminders.append(rem)
            except Exception as e:
                logger.error("Invalid trigger_time in reminder %s: %s", rem.get("id"), str(e))

        return due_reminders

    def mark_completed(self, reminder_id: str) -> bool:
        """Marks a reminder status as completed."""
        reminders = self._read_from_file()
        updated = False

        for rem in reminders:
            if str(rem.get("id")) == str(reminder_id):
                rem["status"] = "completed"
                rem["completed"] = True
                updated = True
                break

        if updated:
            self._write_to_file(reminders)
            logger.info("Marked reminder ID %s as completed.", reminder_id)

        return updated

    def cancel_reminder(self, reminder_id: str) -> bool:
        """Marks a reminder status as cancelled."""
        reminders = self._read_from_file()
        updated = False

        for rem in reminders:
            if str(rem.get("id")) == str(reminder_id):
                rem["status"] = "cancelled"
                rem["completed"] = False
                updated = True
                break

        if updated:
            self._write_to_file(reminders)
            logger.info("Marked reminder ID %s as cancelled.", reminder_id)

        return updated

    def get_all_reminders(self) -> List[Dict[str, Any]]:
        """Returns list of all reminders."""
        return self._read_from_file()

    def get_pending_reminders(self) -> List[Dict[str, Any]]:
        """Returns list of pending reminders."""
        reminders = self._read_from_file()
        return [r for r in reminders if r.get("status", "pending") == "pending" or r.get("completed") is False]

    def delete_reminder(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Cancels a reminder matching description or UUID (sets status to 'cancelled').
        """
        if not query:
            return None

        clean_query = query.strip().lower()
        reminders = self._read_from_file()
        deleted_rem = None

        for rem in reminders:
            rem_id = str(rem.get("id", "")).lower()
            rem_msg = str(rem.get("message", "")).lower()
            rem_text = str(rem.get("text", "")).lower()

            if clean_query == rem_id or clean_query in rem_msg or rem_msg in clean_query or clean_query in rem_text or rem_text in clean_query:
                rem["status"] = "cancelled"
                rem["completed"] = False
                deleted_rem = rem
                break

        if deleted_rem:
            self._write_to_file(reminders)
            logger.info("Cancelled/Deleted reminder: '%s'", deleted_rem.get("message"))
            return deleted_rem

        return None

"""
Morphix Desktop Companion Robot - Local Memory Manager
======================================================
File: src/database/memory.py
Description:
    Provides simple, JSON-based persistent storage for local user memory
    and key-value data for the Morphix desktop companion robot.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional


class Memory:
    """
    Manages local persistent key-value memory storage backed by a JSON file.
    """

    def __init__(self, file_path: str = "memory.json") -> None:
        """
        Initializes the Memory instance and sets up the JSON storage file.

        :param file_path: Path to the JSON storage file (default: "memory.json").
        """
        self.file_path: Path = Path(file_path)

        # Ensure parent directories exist if a nested path is provided
        if self.file_path.parent:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)

        self._ensure_file_exists()

    def _ensure_file_exists(self) -> None:
        """
        Creates the JSON file with an empty dictionary if it doesn't exist.
        Handles empty or corrupted files gracefully by resetting them.
        """
        if not self.file_path.exists():
            self._write_to_file({})
            return

        # Verify existing file content for valid JSON data
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    self._write_to_file({})
        except (json.JSONDecodeError, OSError):
            # Reset file gracefully if corrupted or empty
            self._write_to_file({})

    def _read_from_file(self) -> Dict[str, Any]:
        """
        Reads and returns all stored data from the JSON file.

        :return: Dictionary containing stored key-value pairs.
        """
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
                return {}
        except (json.JSONDecodeError, OSError):
            return {}

    def _write_to_file(self, data: Dict[str, Any]) -> bool:
        """
        Writes a dictionary of data to the JSON storage file.

        :param data: Dictionary to write to the file.
        :return: True if writing succeeded, False otherwise.
        """
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            return True
        except OSError:
            return False

    def save(self, key: str, value: Any) -> bool:
        """
        Saves a key-value pair into local memory.

        :param key: Key name string.
        :param value: Value to store.
        :return: True if saved successfully, False otherwise.
        """
        data = self._read_from_file()
        data[key] = value
        return self._write_to_file(data)

    def load(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Retrieves a stored value associated with the given key.

        :param key: Key name to lookup.
        :param default: Value to return if key is not found (default: None).
        :return: Stored value or default value.
        """
        data = self._read_from_file()
        return data.get(key, default)

    def update(self, key: str, value: Any) -> bool:
        """
        Updates the value of an existing key, or creates it if not present.

        :param key: Key name string.
        :param value: New value to set.
        :return: True if updated successfully, False otherwise.
        """
        return self.save(key, value)

    def delete(self, key: str) -> bool:
        """
        Deletes a key-value pair from local memory.

        :param key: Key name to remove.
        :return: True if deleted successfully, False if key was not found.
        """
        data = self._read_from_file()
        if key in data:
            del data[key]
            self._write_to_file(data)
            return True
        return False

    def exists(self, key: str) -> bool:
        """
        Checks if a key exists in local memory.

        :param key: Key name to check.
        :return: True if key exists, False otherwise.
        """
        data = self._read_from_file()
        return key in data

    def clear(self) -> bool:
        """
        Clears all stored key-value data from local memory.

        :return: True if memory was cleared, False otherwise.
        """
        return self._write_to_file({})

    def get_all(self) -> Dict[str, Any]:
        """
        Returns a copy of all stored key-value pairs in local memory.

        :return: Dictionary containing all memory entries.
        """
        return self._read_from_file()

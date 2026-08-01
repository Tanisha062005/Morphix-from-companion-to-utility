"""
Morphix Desktop Companion Robot - Offline To-Do Models
======================================================
File: src/todo/todo_models.py
Description:
    Represents structures and data classes for To-Do item objects.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TodoItem:
    """
    Data class representing structured properties of a task.
    """
    id: str
    task: str
    created_at: str
    completed: bool
    completed_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> dict:
        """Converts model properties into a dictionary."""
        return {
            "id": self.id,
            "task": self.task,
            "created_at": self.created_at,
            "completed": self.completed,
            "completed_at": self.completed_at,
            "updated_at": self.updated_at
        }

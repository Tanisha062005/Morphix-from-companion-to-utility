"""
Morphix Desktop Companion Robot - Gemini Chat Memory
====================================================
File: src/gemini/chat_memory.py
Description:
    Manages lightweight rolling memory for general AI brain conversation.
"""

from typing import Any, Dict, List
from src.gemini.gemini_config import GeminiConfig


class ChatMemory:
    """
    Maintains a rolling limit of recent exchanges to optimize tokens.
    """

    def __init__(self, limit: int = GeminiConfig.CHAT_HISTORY_LIMIT) -> None:
        self.limit = limit
        self.history: List[Dict[str, Any]] = []

    def add_exchange(self, user_msg: str, assistant_response: str) -> None:
        """Adds a message exchange to the rolling memory."""
        self.history.append({"role": "user", "parts": [user_msg]})
        self.history.append({"role": "model", "parts": [assistant_response]})

        # Prune older messages if history exceeds configured limit
        if len(self.history) > self.limit:
            self.history = self.history[-self.limit:]

    def get_history(self) -> List[Dict[str, Any]]:
        """Returns current history list."""
        return self.history

    def clear(self) -> None:
        """Resets conversation history."""
        self.history.clear()

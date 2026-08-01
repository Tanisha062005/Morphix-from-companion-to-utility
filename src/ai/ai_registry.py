"""
Morphix Desktop Companion Robot - AI Capability Registry
=========================================================
File: src/ai/ai_registry.py
Description:
    Provides a centralized registry for AI capability providers using lazy loading
    to optimize initialization and memory usage.

    Current Providers:
    - Gemini

    Future Providers:
    - OpenAI
    - Claude
    - Local Llama
    - Ollama
    - Vision AI
    - OCR
    - Web Search
"""

import logging
import threading
from typing import Any, Callable, Dict

logger = logging.getLogger("Morphix.AIRegistry")


class AIRegistry:
    """
    Registry for AI providers that uses lazy initialization.
    Saves memory overhead by not instantiating heavyweight models until requested.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(AIRegistry, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._factories: Dict[str, Callable[[], Any]] = {}
        self._instances: Dict[str, Any] = {}
        self._supported_intents = {
            "ADD_TODO", "DELETE_TODO", "EDIT_TODO", "COMPLETE_TODO", "SHOW_TODO",
            "SET_REMINDER", "DELETE_REMINDER", "SHOW_REMINDERS", "CHAT", "UNKNOWN"
        }

    def register_intent(self, intent: str) -> None:
        """Dynamically registers a supported intent."""
        self._supported_intents.add(intent)
        logger.info("Registered dynamic intent: '%s'", intent)

    def get_supported_intents(self) -> set:
        """Returns the set of all currently supported intents."""
        return self._supported_intents

    def register_provider(self, name: str, factory_func: Callable[[], Any]) -> None:
        """
        Registers an AI provider's builder function.

        :param name: Provider identification string.
        :param factory_func: Factory callable returning provider instance.
        """
        self._factories[name] = factory_func
        logger.info("Registered AI provider: '%s' (Lazy Loading Enabled)", name)

    def get_provider(self, name: str) -> Any:
        """
        Retrieves the provider instance, lazy loading if not yet created.

        :param name: Registered provider name.
        :return: Initialized provider instance.
        :raises ValueError: If provider is not registered.
        """
        if name not in self._instances:
            if name in self._factories:
                logger.info("Lazy loading AI provider: '%s'...", name)
                self._instances[name] = self._factories[name]()
            else:
                logger.error("AI provider '%s' is not registered.", name)
                raise ValueError(f"Provider '{name}' is not registered.")
        return self._instances[name]

    def has_provider(self, name: str) -> bool:
        """Checks if a provider name is registered."""
        return name in self._factories

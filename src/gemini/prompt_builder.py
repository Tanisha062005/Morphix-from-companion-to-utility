"""
Morphix Desktop Companion Robot - Gemini Prompt Builder
======================================================
File: src/gemini/prompt_builder.py
Description:
    Imports prompt variants from prompt_versions.py and constructs
    system instructions for NLU and chat tasks.
"""

from src.gemini.prompt_versions import PROMPT_CHAT_V1, PROMPT_INTENT_V1


class PromptBuilder:
    """
    Exposes retrieval methods for the system prompts.
    """

    @staticmethod
    def get_intent_system_prompt() -> str:
        """Returns the system prompt for structured intent extraction."""
        return PROMPT_INTENT_V1

    @staticmethod
    def get_chat_system_prompt() -> str:
        """Returns the system prompt for general conversation."""
        return PROMPT_CHAT_V1

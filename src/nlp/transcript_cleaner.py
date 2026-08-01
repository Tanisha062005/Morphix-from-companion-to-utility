"""
Morphix Desktop Companion Robot - Speech Transcript Cleaner
===========================================================
File: src/nlp/transcript_cleaner.py
Description:
    Provides normalization routines for Speech-To-Text transcripts.
    Converts casing to lowercase, removes punctuation, normalizes spacing,
    and translates number words (e.g., 'two' -> '2') for downstream parsing.
"""

import re


class TranscriptCleaner:
    """
    Utility class to clean and normalize raw speech transcription outputs.
    """

    NUMBER_WORDS = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
        "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
        "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
        "forty": "40", "fifty": "50", "sixty": "60"
    }

    @classmethod
    def clean(cls, text: str) -> str:
        """
        Normalizes spacing, casing, punctuation, and written numbers.

        :param text: Raw transcript string.
        :return: Normalized transcript.
        """
        if not text or not isinstance(text, str):
            return ""

        # 1. Lowercase conversion
        cleaned = text.lower()

        # 2. Punctuation normalization (replace with spaces)
        cleaned = re.sub(r"[^\w\s-]", " ", cleaned)

        # 3. Replace common written phrases (e.g., 'to do' -> 'todo')
        cleaned = re.sub(r"\bto\s+do\b", "todo", cleaned)

        # 4. Translate number words to digits
        words = cleaned.split()
        normalized_words = []
        for word in words:
            if word in cls.NUMBER_WORDS:
                normalized_words.append(cls.NUMBER_WORDS[word])
            else:
                normalized_words.append(word)

        cleaned = " ".join(normalized_words)

        # 5. Remove duplicate spaces and strip surrounding whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned

"""
Morphix Desktop Companion Robot - Speech Transcript Cleaner
===========================================================
File: src/nlp/transcript_cleaner.py
Description:
    Provides normalization routines for Speech-To-Text transcripts.
    Converts casing to lowercase, removes punctuation, normalizes spacing,
    translates number words, and maps Google STT phonetic misrecognitions to 'morphix'.
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

    DEVANAGARI_MAPS = {
        "रिमाइंड": "remind",
        "रिमाइंडर": "remind",
        "ऐड": "add",
        "टूडू": "todo",
        "टू-डू": "todo"
    }

    # Common English contraction expansions applied BEFORE punctuation stripping.
    CONTRACTION_MAP = {
        r"\bi'm\b": "i am",
        r"\bi've\b": "i have",
        r"\bi'll\b": "i will",
        r"\bi'd\b": "i would",
        r"\bdon't\b": "do not",
        r"\bdoesn't\b": "does not",
        r"\bdidn't\b": "did not",
        r"\bcan't\b": "cannot",
        r"\bcouldn't\b": "could not",
        r"\bwon't\b": "will not",
        r"\bwouldn't\b": "would not",
        r"\bshouldn't\b": "should not",
        r"\bisn't\b": "is not",
        r"\baren't\b": "are not",
        r"\bwasn't\b": "was not",
        r"\bweren't\b": "were not",
        r"\bhaven't\b": "have not",
        r"\bhadn't\b": "had not",
        r"\bhasn't\b": "has not",
        r"\bit's\b": "it is",
        r"\bthat's\b": "that is",
        r"\bhe's\b": "he is",
        r"\bshe's\b": "she is",
        r"\bwe're\b": "we are",
        r"\bthey're\b": "they are",
        r"\byou're\b": "you are",
        r"\blet's\b": "let us",
    }

    # Google STT Phonetic misrecognition rules mapping to 'morphix'
    PHONETIC_STT_MAP = {
        r"\bmorphing\b": "morphix",
        r"\bmove\s+fix\b": "morphix",
        r"\bmoments\b": "morphix",
        r"\bmorfix\b": "morphix",
        r"\bmore\s+pics\b": "morphix",
        r"\bmore\s+fix\b": "morphix",
        r"\bmorefiks\b": "morphix",
        r"\bmorphics\b": "morphix",
    }

    @classmethod
    def clean(cls, text: str) -> str:
        """
        Normalizes spacing, casing, punctuation, written numbers, and STT phonetics.
        Expands contractions before stripping punctuation to preserve
        fast-path keyword semantics (e.g. "I'm back" → "i am back").

        :param text: Raw transcript string.
        :return: Normalized transcript.
        """
        if not text or not isinstance(text, str):
            return ""

        # 1. Lowercase conversion
        cleaned = text.lower()

        # 2. Expand contractions BEFORE stripping punctuation
        for pattern, expansion in cls.CONTRACTION_MAP.items():
            cleaned = re.sub(pattern, expansion, cleaned)

        # 3. Phonetic STT misrecognition normalization (e.g. "move fix" -> "morphix")
        for pattern, replacement in cls.PHONETIC_STT_MAP.items():
            cleaned = re.sub(pattern, replacement, cleaned)

        # 4. Punctuation normalization (replace with spaces)
        cleaned = re.sub(r"[.,\/#!$%\^&\*;:{}=\-_`~()?'\"]", " ", cleaned)

        # 5. Replace common written phrases (e.g., 'to do' -> 'todo')
        cleaned = re.sub(r"\bto\s+do\b", "todo", cleaned)

        # 6. Translate number words to digits and Devanagari command words to English
        words = cleaned.split()
        normalized_words = []
        for word in words:
            if word in cls.DEVANAGARI_MAPS:
                normalized_words.append(cls.DEVANAGARI_MAPS[word])
            elif word in cls.NUMBER_WORDS:
                normalized_words.append(cls.NUMBER_WORDS[word])
            else:
                normalized_words.append(word)

        cleaned = " ".join(normalized_words)

        # 7. Remove duplicate spaces and strip surrounding whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned

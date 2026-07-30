"""
Morphix Desktop Companion Robot - Wake Word Detector
=====================================================
File: src/voice/wake_word.py
Description:
    Provides intelligent multi-algorithm similarity wake-word detection for Morphix.
    Normalizes input text and computes fuzzy similarity against multiple target wake phrases
    and configured accepted phonetic aliases.
"""

import difflib
import itertools
import logging
import re
import string
import sys
from typing import List, Optional, Tuple, Union

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

from config import Config

# Setup logger for WakeWordDetector module
logger = logging.getLogger("Morphix.WakeWord")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [WakeWord]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class WakeWordDetector:
    """
    Manages wake word detection logic and listening status for Morphix.
    Performs similarity matching against multiple target wake phrases and aliases.
    """

    def __init__(
        self,
        wake_phrases: Optional[List[str]] = None,
        threshold: Optional[float] = None,
        wake_word: Optional[str] = None,
        aliases: Optional[List[str]] = None
    ) -> None:
        """
        Initializes WakeWordDetector. Supports multiple wake phrases, maintaining backward compatibility.

        :param wake_phrases: Configurable list of target wake words (default: Config.WAKE_PHRASES).
        :param threshold: Minimum similarity threshold (default: Config.SIMILARITY_THRESHOLD).
        :param wake_word: Backward compatibility fallback for primary wake word.
        :param aliases: Backward compatibility fallback for phonetic aliases.
        """
        raw_thresh = threshold if threshold is not None else getattr(Config, "SIMILARITY_THRESHOLD", 0.85)
        # Normalize internal threshold to 0.0 - 100.0 scale
        self._threshold: float = raw_thresh * 100.0 if raw_thresh <= 1.0 else float(raw_thresh)

        if wake_phrases is not None:
            self._wake_phrases: List[str] = list(wake_phrases)
        else:
            phrases = []
            if wake_word:
                phrases.append(wake_word)
            else:
                phrases.append(getattr(Config, "WAKE_PHRASE", "hey morphix"))

            if aliases is not None:
                phrases.extend(aliases)
            else:
                phrases.extend(getattr(Config, "ACCEPTED_ALIASES", []))

            # Deduplicate while preserving order
            unique_phrases = []
            for p in phrases:
                if p not in unique_phrases:
                    unique_phrases.append(p)
            self._wake_phrases = unique_phrases

        self._primary_phrases = ["hey morphix", "hello morphix", "hi morphix", "morphix", "wake up morphix"]
        self._fallback_aliases = []
        for p in self._wake_phrases:
            norm_p = self.normalize_text(p)
            if norm_p not in self._primary_phrases:
                if p not in self._fallback_aliases:
                    self._fallback_aliases.append(p)

        self._listening: bool = False
        self._primary_matched: str = "NO"
        self._alias_matched: str = "NO"
        self._fuzzy_required: str = "NO"

    def start_listening(self) -> None:
        """Enables listening state."""
        self._listening = True

    def stop_listening(self) -> None:
        """Disables listening state."""
        self._listening = False

    def is_listening(self) -> bool:
        """Returns whether detector is currently listening."""
        return self._listening

    def get_wake_word(self) -> str:
        """Returns the primary target wake word."""
        return self._wake_phrases[0] if self._wake_phrases else "hey morphix"

    @staticmethod
    def _is_sublist(sublist: List[str], main_list: List[str]) -> bool:
        if not sublist:
            return False
        sub_len = len(sublist)
        main_len = len(main_list)
        for i in range(main_len - sub_len + 1):
            if main_list[i:i+sub_len] == sublist:
                return True
        return False

    def _get_word_similarity(self, w1: str, w2: str) -> float:
        if len(w1) == 1 or len(w2) == 1:
            return 100.0 if w1 == w2 else 0.0
        if HAS_RAPIDFUZZ:
            sim = float(fuzz.ratio(w1, w2))
        else:
            sim = difflib.SequenceMatcher(None, w1, w2).ratio() * 100.0
        return sim

    def _is_last_token_match(self, w_last: str, p_last: str) -> bool:
        sim = self._get_word_similarity(w_last, p_last)
        if sim >= 50.0:
            return True
        if w_last and p_last and w_last[0] == p_last[0] and abs(len(w_last) - len(p_last)) <= 1:
            return True
        return False

    def _is_valid_fuzzy_match(self, window_tokens: List[str], phrase_tokens: List[str], input_tokens: List[str]) -> bool:
        # 1. Single token protection
        if len(input_tokens) == 1 and len(phrase_tokens) > 1:
            return False

        # 2. Single character protection
        for tok in window_tokens:
            if len(tok) == 1:
                if tok not in phrase_tokens:
                    return False

        # 3. Last token match validation
        if window_tokens and phrase_tokens:
            if not self._is_last_token_match(window_tokens[-1], phrase_tokens[-1]):
                return False

        return True

    def _get_weighted_fuzzy_similarity(self, window_tokens: List[str], phrase_tokens: List[str]) -> float:
        M, N = len(window_tokens), len(phrase_tokens)
        if M == 0 or N == 0:
            return 0.0

        FILLER_WORDS = {"hey", "hello", "hi", "he", "you", "wake", "up", "and", "no", "is", "the", "on", "am", "come", "game"}
        weights = [0.5 if t in FILLER_WORDS else 1.0 for t in phrase_tokens]

        if M == N:
            total_sim = 0.0
            total_weight = sum(weights)
            for i in range(N):
                sim = self._get_word_similarity(window_tokens[i], phrase_tokens[i])
                total_sim += sim * weights[i]
            return total_sim / total_weight if total_weight > 0 else 0.0

        elif M < N:
            best_score = 0.0
            for gap_idx in range(N):
                total_sim = 0.0
                total_weight = sum(weights)
                for i in range(N):
                    if i < gap_idx:
                        sim = self._get_word_similarity(window_tokens[i], phrase_tokens[i])
                    elif i == gap_idx:
                        sim = 0.0
                    else:
                        sim = self._get_word_similarity(window_tokens[i-1], phrase_tokens[i])
                    total_sim += sim * weights[i]
                score = total_sim / total_weight if total_weight > 0 else 0.0
                if score > best_score:
                    best_score = score
            return best_score

        else: # M > N
            best_score = 0.0
            for gap_idx in range(M):
                total_sim = 0.0
                b_idx = 0
                total_weight = 0.0
                for i in range(M):
                    if i == gap_idx:
                        continue
                    sim = self._get_word_similarity(window_tokens[i], phrase_tokens[b_idx])
                    w = weights[b_idx]
                    total_sim += sim * w
                    total_weight += w
                    b_idx += 1
                score = total_sim / total_weight if total_weight > 0 else 0.0
                if score > best_score:
                    best_score = score
            return best_score

    def _get_non_filler_fuzzy_similarity(self, window_tokens: List[str], phrase_tokens: List[str]) -> float:
        FILLER_WORDS = {"hey", "hello", "hi", "he", "you", "wake", "up", "and", "no", "is", "the", "on", "am", "come", "game"}
        win_non_fill = [t for t in window_tokens if t not in FILLER_WORDS]
        phrase_non_fill = [t for t in phrase_tokens if t not in FILLER_WORDS]

        if not phrase_non_fill or not win_non_fill:
            return 0.0

        M, N = len(win_non_fill), len(phrase_non_fill)
        if M == N:
            total_sim = 0.0
            for i in range(N):
                total_sim += self._get_word_similarity(win_non_fill[i], phrase_non_fill[i])
            return total_sim / N
        elif M < N:
            best_score = 0.0
            for gap_idx in range(N):
                total_sim = 0.0
                for i in range(N):
                    if i < gap_idx:
                        sim = self._get_word_similarity(win_non_fill[i], phrase_non_fill[i])
                    elif i == gap_idx:
                        sim = 0.0
                    else:
                        sim = self._get_word_similarity(win_non_fill[i-1], phrase_non_fill[i])
                    total_sim += sim
                score = total_sim / N
                if score > best_score:
                    best_score = score
            return best_score
        else: # M > N
            best_score = 0.0
            for indices in itertools.combinations(range(M), N):
                total_sim = 0.0
                for idx_i, win_idx in enumerate(indices):
                    total_sim += self._get_word_similarity(win_non_fill[win_idx], phrase_non_fill[idx_i])
                score = total_sim / N
                if score > best_score:
                    best_score = score
            return best_score

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalizes input text: lowercases, strips [BLANK_AUDIO] tokens, punctuation, and spaces.
        """
        if not text or not isinstance(text, str):
            return ""

        # 1. Remove [BLANK_AUDIO] tokens case-insensitively before matching
        cleaned = re.sub(r'(?i)\[BLANK_AUDIO\]', '', text)

        # 2. Convert to lowercase
        lowercased = cleaned.lower()

        # 3. Remove punctuation
        no_punct = lowercased.translate(str.maketrans("", "", string.punctuation))

        # 4. Collapse whitespace
        collapsed = re.sub(r"\s+", " ", no_punct).strip()
        return collapsed

    def compute_similarity(self, text: str) -> Tuple[float, str, str, float, bool]:
        """
        Computes similarity for input text against target wake phrases.

        :param text: Raw input speech text.
        :return: Tuple of (best_score, matched_wake_phrase, matched_alias, similarity, detection_result).
        """
        norm_input = self.normalize_text(text)
        if not norm_input:
            self._primary_matched = "NO"
            self._alias_matched = "NO"
            self._fuzzy_required = "NO"
            return 0.0, self.get_wake_word(), "", 0.0, False

        input_tokens = norm_input.split()

        self._primary_matched = "NO"
        self._alias_matched = "NO"
        self._fuzzy_required = "NO"

        # 1. Primary phrase substring match
        for phrase in self._primary_phrases:
            norm_phrase = self.normalize_text(phrase)
            if not norm_phrase:
                continue
            if norm_phrase in norm_input:
                self._primary_matched = "YES"
                return 100.0, phrase, norm_phrase, 1.0, True

        # 2. Alias substring match
        for phrase in self._fallback_aliases:
            norm_phrase = self.normalize_text(phrase)
            if not norm_phrase:
                continue
            if norm_phrase in norm_input:
                self._alias_matched = "YES"
                return 100.0, phrase, norm_phrase, 1.0, True

        self._fuzzy_required = "YES"

        # 3. Fuzzy Matching
        best_score = 0.0
        best_phrase = self.get_wake_word()

        for phrase in self._wake_phrases:
            norm_phrase = self.normalize_text(phrase)
            if not norm_phrase:
                continue
            phrase_tokens = norm_phrase.split()
            N = len(phrase_tokens)
            phrase_best_score = 0.0
            for w_len in range(max(1, N - 1), N + 2):
                if w_len > len(input_tokens):
                    continue
                for i in range(len(input_tokens) - w_len + 1):
                    window_tokens = input_tokens[i:i+w_len]
                    if not self._is_valid_fuzzy_match(window_tokens, phrase_tokens, input_tokens):
                        continue

                    s_weighted = self._get_weighted_fuzzy_similarity(window_tokens, phrase_tokens)
                    s_non_filler = self._get_non_filler_fuzzy_similarity(window_tokens, phrase_tokens)
                    score = max(s_weighted, s_non_filler)

                    if score > phrase_best_score:
                        phrase_best_score = score

            if phrase_best_score > best_score:
                best_score = phrase_best_score
                best_phrase = phrase

        is_detected = best_score >= self._threshold
        similarity = round(best_score / 100.0, 2)
        best_score_rounded = round(best_score, 1)
        matched_alias = self.normalize_text(best_phrase)

        return best_score_rounded, best_phrase, matched_alias, similarity, is_detected

    def detect(self, text: str) -> bool:
        """
        Evaluates input text against target wake phrases.
        Prints and logs exact required formats.
        """
        if not text or not isinstance(text, str):
            return False

        raw_text = text.strip()
        best_score, matched_phrase, matched_alias, similarity, is_wake = self.compute_similarity(raw_text)

        status_str = "YES" if is_wake else "NO"
        primary_match_str = "YES" if self._primary_matched == "YES" else "NO"
        alias_match_str = "YES" if self._alias_matched == "YES" else "NO"
        fuzzy_used_str = "YES" if self._fuzzy_required == "YES" else "NO"
        threshold_display = self._threshold / 100.0

        # Detailed debugging information printed directly as requested
        print("\nNormalized Transcript:", self.normalize_text(raw_text))
        print("Primary Phrase Match:", primary_match_str)
        print("Alias Phrase Match:", alias_match_str)
        print("Best Matching Phrase:", matched_phrase)
        print("Best Similarity Score:", f"{similarity:.2f}")
        print("Fuzzy Matching Used:", fuzzy_used_str)
        print("Wake Detected:", status_str)

        # Direct exact print format as requested for compatibility
        print("\nRaw:")
        print(f'"{raw_text}"\n')
        print("Normalized:")
        print(f'"{self.normalize_text(raw_text)}"\n')
        print("Similarity:")
        print(f"{similarity:.2f}\n")
        print("Matched Variant:")
        print(f'"{matched_phrase}"\n')
        print("Threshold:")
        print(f"{threshold_display:.2f}\n")
        print("Wake Detected:")
        print(f"{status_str}\n")

        # Logging output for auditability
        logger.info("Raw Speech: %s", raw_text)
        logger.info("Normalized Speech: %s", self.normalize_text(raw_text))
        logger.info("Similarity Score: %.2f", similarity)
        logger.info("Matched Variant (Wake Phrase): %s", matched_phrase)
        logger.info("Matched Alias: %s", matched_alias)
        logger.info("Threshold: %.2f", threshold_display)
        logger.info("Wake Detected: %s", status_str)

        return is_wake


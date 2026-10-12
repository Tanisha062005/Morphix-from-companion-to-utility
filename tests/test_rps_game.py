"""
Unit Tests for Rock-Paper-Scissors (RPS) Game Plugin & Vision Subsystems
========================================================================
File: tests/test_rps_game.py
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.intent.intent_classifier import IntentClassifier, IntentResult, IntentType
from src.intent.rps_game_plugin import RPSGamePlugin, RPSService
from src.vision.gesture_detector import GestureDetector


class TestRPSLogic(unittest.TestCase):
    """Tests RPS game business logic and winner evaluation rules."""

    def setUp(self):
        self.service = RPSService.get_instance()

    def test_decide_winner_player_wins(self):
        self.assertEqual(self.service.decide_winner("ROCK", "SCISSORS"), "PLAYER")
        self.assertEqual(self.service.decide_winner("SCISSORS", "PAPER"), "PLAYER")
        self.assertEqual(self.service.decide_winner("PAPER", "ROCK"), "PLAYER")

    def test_decide_winner_morphix_wins(self):
        self.assertEqual(self.service.decide_winner("SCISSORS", "ROCK"), "MORPHIX")
        self.assertEqual(self.service.decide_winner("PAPER", "SCISSORS"), "MORPHIX")
        self.assertEqual(self.service.decide_winner("ROCK", "PAPER"), "MORPHIX")

    def test_decide_winner_ties(self):
        self.assertEqual(self.service.decide_winner("ROCK", "ROCK"), "TIE")
        self.assertEqual(self.service.decide_winner("PAPER", "PAPER"), "TIE")
        self.assertEqual(self.service.decide_winner("SCISSORS", "SCISSORS"), "TIE")

    def test_decide_winner_unknown_input(self):
        self.assertEqual(self.service.decide_winner("UNKNOWN", "ROCK"), "UNKNOWN")
        self.assertEqual(self.service.decide_winner("", "SCISSORS"), "UNKNOWN")
        self.assertEqual(self.service.decide_winner("INVALID", "PAPER"), "UNKNOWN")


class TestRPSIntentClassification(unittest.TestCase):
    """Tests intent classification for English and Hinglish RPS phrases."""

    def setUp(self):
        self.classifier = IntentClassifier()

    def test_rps_intent_triggers(self):
        test_phrases = [
            "play rock paper scissors",
            "let's play rock paper scissors",
            "rock paper scissors khelo",
            "stone paper scissor",
            "khelte hain rock paper scissor",
            "rock paper scissors",
            "khelo rock paper scissors",
            "play rps",
            "lets play rps",
            "chalo rock paper scissors khelo",
        ]
        for phrase in test_phrases:
            res = self.classifier.classify(phrase)
            self.assertEqual(
                res.intent,
                IntentType.PLAY_RPS,
                f"Phrase '{phrase}' should classify as PLAY_RPS, got {res.intent}"
            )

    def test_rps_plugin_direct_match(self):
        plugin = RPSGamePlugin()
        res = plugin.match("play rock paper scissors")
        self.assertIsNotNone(res)
        self.assertEqual(res.intent, IntentType.PLAY_RPS)

        res_hi = plugin.match("rock paper scissors khelo")
        self.assertIsNotNone(res_hi)
        self.assertTrue(res_hi.parameters.get("is_hindi", False))

    def test_rps_plugin_handle(self):
        plugin = RPSGamePlugin()
        mock_disp = MagicMock()
        mock_tts = MagicMock()
        res = IntentResult(intent=IntentType.PLAY_RPS, parameters={"is_hindi": False})

        with patch.object(RPSService, "capture_frame_safely", return_value=None):
            resp = plugin.handle(res, display=mock_disp, tts=mock_tts)
            self.assertIsInstance(resp, str)
            self.assertNotEqual(resp, "I'm not sure how to do that yet.")
            self.assertTrue(len(resp) > 0)


class TestGestureDetector(unittest.TestCase):
    """Tests vision gesture detector edge cases and classification."""

    def setUp(self):
        self.detector = GestureDetector.get_instance()

    def test_empty_or_none_frame(self):
        gesture, conf, details = self.detector.detect_gesture(None)
        self.assertEqual(gesture, "UNKNOWN")

        empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
        gesture, conf, details = self.detector.detect_gesture(empty_frame)
        self.assertEqual(gesture, "UNKNOWN")

    def test_blank_black_frame(self):
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        gesture, conf, details = self.detector.detect_gesture(black_frame)
        self.assertEqual(gesture, "UNKNOWN")


class TestRPSDisplayOverlay(unittest.TestCase):
    """Tests Pygame display overlays for countdown and result states."""

    def test_eye_animator_rps_overlays(self):
        import pygame
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        surface = pygame.Surface((480, 240))

        from src.display.eye_animator import EyeAnimator

        animator = EyeAnimator(480, 240)

        # Test RPS_COUNTDOWN overlay
        animator.set_overlay("RPS_COUNTDOWN", data={"count": "3"}, timeout=3.0)
        self.assertEqual(animator.active_overlay, "RPS_COUNTDOWN")
        animator.update(0.1)
        animator.draw(surface)

        # Test RPS_RESULT overlay
        animator.set_overlay("RPS_RESULT", data={"player": "ROCK", "morphix": "SCISSORS", "winner": "PLAYER"}, timeout=5.0)
        self.assertEqual(animator.active_overlay, "RPS_RESULT")
        animator.update(0.1)
        animator.draw(surface)

        # Test Tie result
        animator.show_rps_result("PAPER", "PAPER", "TIE")
        self.assertEqual(animator.active_overlay, "RPS_RESULT")
        animator.draw(surface)

        # Test clear overlay
        animator.clear_overlay()
        self.assertIsNone(animator.active_overlay)
        animator.draw(surface)

        pygame.quit()


if __name__ == "__main__":
    unittest.main()

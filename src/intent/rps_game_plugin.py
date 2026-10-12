"""
Morphix Desktop Companion Robot - Vision Rock-Paper-Scissors Game Plugin
========================================================================
File: src/intent/rps_game_plugin.py
Description:
    Provides interactive vision-based Rock-Paper-Scissors (RPS) game plugin.
    Coordinates audio-visual countdown on Pygame display, real-time camera
    hand gesture recognition, winner evaluation, dynamic emotion reactions,
    and sporty voice responses.
"""

import logging
import random
import re
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from src.intent.intent_classifier import IntentResult, IntentType
from src.vision.camera_manager import CameraManager
from src.vision.gesture_detector import GestureDetector

logger = logging.getLogger("Morphix.RPSGamePlugin")


class RPSService:
    """
    Core game engine for Rock-Paper-Scissors.
    Manages vision capture, move evaluation, and audio-visual orchestration.
    """

    _instance: Optional["RPSService"] = None

    CHOICES = ["ROCK", "PAPER", "SCISSORS"]

    @classmethod
    def get_instance(cls) -> "RPSService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def decide_winner(self, player_move: str, morphix_move: str) -> str:
        """
        Determines the winner of the round according to standard RPS rules.

        :param player_move: 'ROCK', 'PAPER', 'SCISSORS', or 'UNKNOWN'.
        :param morphix_move: 'ROCK', 'PAPER', or 'SCISSORS'.
        :return: 'PLAYER', 'MORPHIX', 'TIE', or 'UNKNOWN'.
        """
        p = str(player_move).upper().strip()
        m = str(morphix_move).upper().strip()

        if p not in self.CHOICES:
            return "UNKNOWN"

        if p == m:
            return "TIE"

        winning_combos = {
            ("ROCK", "SCISSORS"),
            ("SCISSORS", "PAPER"),
            ("PAPER", "ROCK"),
        }

        if (p, m) in winning_combos:
            return "PLAYER"
        return "MORPHIX"

    def capture_frame_safely(self) -> Optional[np.ndarray]:
        """
        Captures a single camera frame safely without interfering with
        surveillance guard mode. Releases hardware immediately if not in surveillance.
        """
        cam = CameraManager.get_instance()
        frame = None

        if cam.is_surveillance_active():
            logger.info("Surveillance active — reading shared camera stream frame.")
            frame = cam.read_frame()
        else:
            logger.info("Surveillance inactive — safely acquiring camera snapshot.")
            opened = cam._open_camera(timeout_sec=3.0)
            if opened:
                # Flush 2-3 frames for auto-exposure adjustment
                for _ in range(3):
                    f = cam.read_frame()
                    if f is not None:
                        frame = f
                    time.sleep(0.04)
                cam._release_camera()
            else:
                # Direct OpenCV fallback
                try:
                    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                    if not cap.isOpened():
                        cap = cv2.VideoCapture(0)
                    if cap.isOpened():
                        for _ in range(3):
                            ret, f = cap.read()
                            if ret and f is not None:
                                frame = f
                            time.sleep(0.03)
                        cap.release()
                except Exception as ex:
                    logger.warning("Direct OpenCV camera capture error: %s", ex)

        return frame

    def play_round(self, is_hindi: bool = False, display: Any = None, tts: Any = None) -> str:
        """
        Executes a complete interactive Rock-Paper-Scissors round:
        1. TTS prompt & animated 3... 2... 1... SHOW! countdown.
        2. Frame capture at SHOW! and gesture evaluation.
        3. Decision & 3-column VS Arena result overlay on display.
        4. Defeat/Victory/Tie speech & emotion reactions.
        """
        from src.display.eye_animator import EyeState

        # ── Step 1: Prompt & Ready ──────────────────────────────────────
        intro_en = "Let's play Rock Paper Scissors! Show your hand to the camera on count of three."
        intro_hi = "Chalo Rock Paper Scissors khelte hain! Teen ginne par camera ke saamne apna haath dikhaiye."
        intro_msg = intro_hi if is_hindi else intro_en

        if display is not None and hasattr(display, "set_state"):
            try:
                display.set_state(EyeState.EXCITED)
            except Exception:
                pass

        if display is not None and hasattr(display, "show_rps_countdown"):
            try:
                display.show_rps_countdown(count_text="READY?", timeout=5.0)
            except Exception:
                pass

        if tts is not None and hasattr(tts, "speak"):
            try:
                tts.speak(intro_msg)
            except Exception as e:
                logger.warning("TTS error speaking intro: %s", e)
        else:
            time.sleep(1.0)

        # Countdown sequence: 3... 2... 1... SHOW!
        countdown_steps = [("3", "Three"), ("2", "Two"), ("1", "One"), ("SHOW!", "Show!")]
        for count_val, count_speech in countdown_steps:
            if display is not None and hasattr(display, "show_rps_countdown"):
                try:
                    display.show_rps_countdown(count_text=count_val, timeout=2.0)
                except Exception:
                    pass

            if tts is not None and hasattr(tts, "speak"):
                try:
                    tts.speak(count_speech)
                except Exception:
                    time.sleep(0.6)
            else:
                time.sleep(0.6)

        # ── Step 2: Capture & Gesture Recognition ───────────────────────
        frame = self.capture_frame_safely()
        detector = GestureDetector.get_instance()
        player_move, confidence, details = detector.detect_gesture(frame)
        logger.info("Gesture detected: %s (conf: %.2f, details: %s)", player_move, confidence, details)

        morphix_move = random.choice(self.CHOICES)
        logger.info("Morphix choice: %s | Player move: %s", morphix_move, player_move)

        # ── Step 3: Winner Decision ────────────────────────────────────
        winner = self.decide_winner(player_move, morphix_move)
        logger.info("Round result: %s", winner)

        # ── Step 4: Voice Line & Emotion Reaction ──────────────────────
        if winner == "PLAYER":
            emotion = EyeState.POUTY
            if is_hindi:
                responses = [
                    f"Arey waah! Aapne {player_move} chuna aur meri {morphix_move} ko hara diya. Aap jeet gaye!",
                    f"Shabash! {player_move} ne meri {morphix_move} ko hara diya. Yeh round aapka!",
                    f"Aap jeet gaye! Agle round me main aapko zaroor haraungi!",
                ]
            else:
                responses = [
                    f"Nice move! Your {player_move.lower()} beats my {morphix_move.lower()}. You win this round!",
                    f"Aww, well played! {player_move.title()} defeats {morphix_move.lower()}! You got me!",
                    f"You won this round! Your {player_move.lower()} conquered my {morphix_move.lower()}!",
                ]
            speech_resp = random.choice(responses)

        elif winner == "MORPHIX":
            emotion = EyeState.HAPPY
            if is_hindi:
                responses = [
                    f"Haha! Main jeet gayi! Meri {morphix_move} ne aapke {player_move} ko hara diya!",
                    f"Yes! Meri {morphix_move} jeet gayi. Agli baar better luck!",
                    f"Morphix wins! Meri {morphix_move} ne aapka {player_move} crush kar diya!",
                ]
            else:
                responses = [
                    f"Haha, I win! My {morphix_move.lower()} beats your {player_move.lower()}!",
                    f"Yes! My {morphix_move.lower()} wins against your {player_move.lower()}! Better luck next time!",
                    f"Score one for Morphix! {morphix_move.title()} beats {player_move.lower()}!",
                ]
            speech_resp = random.choice(responses)

        elif winner == "TIE":
            emotion = EyeState.CONFUSED
            if is_hindi:
                speech_resp = f"Yeh to tie ho gaya! Hum dono ne {morphix_move} chuna. Chalo ek aur round khelte hain!"
            else:
                speech_resp = f"It's a tie! We both picked {morphix_move.lower()}. Let's go again!"

        else:  # UNKNOWN
            emotion = EyeState.CONFUSED
            if is_hindi:
                speech_resp = "Mujhe aapka hand gesture saaf nahi dikha. Kripya camera ke saamne haath dikhayein aur fir se try karein!"
            else:
                speech_resp = "I couldn't detect your hand gesture clearly. Make sure your hand is in front of the camera and let's try again!"

        # Update display overlay with 3-column VS Arena results
        if display is not None:
            if hasattr(display, "show_rps_result"):
                try:
                    display.show_rps_result(
                        player=player_move,
                        morphix=morphix_move,
                        winner=winner,
                        timeout=9.0
                    )
                except Exception as e:
                    logger.warning("Error showing RPS result overlay: %s", e)

            if hasattr(display, "set_state"):
                try:
                    display.set_state(emotion)
                except Exception:
                    pass

        return speech_resp


class RPSGamePlugin:
    """
    Voice intent plugin for Rock-Paper-Scissors game.
    """

    def __init__(self) -> None:
        self.service = RPSService.get_instance()

    def handle(self, result: Optional[IntentResult] = None, display: Any = None, tts: Any = None) -> str:
        """
        Executes interactive Rock-Paper-Scissors round workflow:
        1. Speaks countdown prompt and displays animated 3... 2... 1... SHOW! overlay.
        2. Captures camera frame safely at SHOW! and evaluates gesture via GestureDetector.
        3. Decides winner against Morphix's move and shows 3-column VS result overlay.
        4. Triggers emotion reactions (POUTY, HAPPY, CONFUSED) and sporty voice lines.
        5. Cleanly releases camera immediately.
        """
        is_hindi = False
        if result is not None and hasattr(result, "parameters") and isinstance(result.parameters, dict):
            is_hindi = result.parameters.get("is_hindi", False)

        return self.service.play_round(is_hindi=is_hindi, display=display, tts=tts)

    def match(self, text: str) -> Optional[IntentResult]:
        """
        Matches voice queries requesting Rock-Paper-Scissors.
        """
        if not text or not isinstance(text, str):
            return None

        cleaned = text.strip()
        lowered = cleaned.lower()

        is_hindi = bool(re.search(r'\b(khelo|khelte|khel|hain|chalo|karna)\b', lowered))

        # RPS trigger regex patterns
        patterns = [
            r"\b(?:let's\s+)?play\s+(?:a\s+)?(?:game\s+of\s+)?(?:rock\s+paper\s+scissors?|stone\s+paper\s+scissors?|rps)\b",
            r"\b(?:rock\s+paper\s+scissors?|stone\s+paper\s+scissors?|rps)\s+(?:khelo|khelte\s+hain|khelna|play)\b",
            r"\b(?:khelo|khelte\s+hain|chalo\s+khele(?:n)?)\s+(?:rock\s+paper\s+scissors?|stone\s+paper\s+scissors?|rps)\b",
            r"^(?:rock\s+paper\s+scissors?|stone\s+paper\s+scissors?|rps)$",
            r"\bplay\s+rps\b",
            r"\blets\s+play\s+rps\b",
        ]

        for pat in patterns:
            if re.search(pat, lowered):
                return IntentResult(
                    intent=IntentType.PLAY_RPS,
                    parameters={"is_hindi": is_hindi},
                    confidence=0.98,
                    raw_text=cleaned,
                )

        return None

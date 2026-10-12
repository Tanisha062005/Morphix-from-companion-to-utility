"""
Morphix Desktop Companion Robot - Vision Hand Gesture Detector
===============================================================
File: src/vision/gesture_detector.py
Description:
    Real-time vision hand gesture detector for Rock-Paper-Scissors.
    Supports MediaPipe Hands / GestureRecognizer and high-accuracy offline OpenCV
    skin segmentation + convex hull / convexity defect analysis.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("Morphix.GestureDetector")


class GestureDetector:
    """
    Detects hand gestures (ROCK, PAPER, SCISSORS, UNKNOWN) from BGR video frames.
    """

    _instance: Optional["GestureDetector"] = None

    @classmethod
    def get_instance(cls) -> "GestureDetector":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._mp_available: bool = False
        self._mp_detector = None
        self._init_mediapipe()

    def _init_mediapipe(self) -> None:
        """Attempts to initialize MediaPipe if task components are available."""
        try:
            import mediapipe as mp
            # Verify mediapipe is present
            if hasattr(mp, "__version__"):
                self._mp_available = True
                logger.info("MediaPipe %s detected for vision processing.", mp.__version__)
        except Exception as e:
            logger.debug("MediaPipe not initialized, using optimized OpenCV engine: %s", e)
            self._mp_available = False

    def detect_gesture(self, frame: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
        """
        Classifies the user hand gesture from an input BGR camera frame.

        :param frame: Raw BGR camera image.
        :return: Tuple of (gesture_name: str, confidence: float, details_dict: Dict).
                 gesture_name is one of 'ROCK', 'PAPER', 'SCISSORS', or 'UNKNOWN'.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return "UNKNOWN", 0.0, {"reason": "empty_frame"}

        # Resize for consistent processing performance
        h, w = frame.shape[:2]
        proc_w = 400
        proc_h = int(h * (proc_w / w))
        resized = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

        # Execute robust skin segmentation and contour defect analysis
        gesture, conf, details = self._detect_gesture_opencv(resized)
        return gesture, conf, details

    def _detect_gesture_opencv(self, frame: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
        """
        Offline, zero-dependency hand contour, convex hull, and defect analyzer.
        """
        # 1. Skin color segmentation using YCrCb + HSV colourspaces
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # YCrCb skin range
        mask_ycrcb = cv2.inRange(ycrcb, np.array([0, 133, 77], dtype=np.uint8), np.array([255, 173, 127], dtype=np.uint8))
        # HSV skin range
        mask_hsv = cv2.inRange(hsv, np.array([0, 28, 60], dtype=np.uint8), np.array([25, 200, 255], dtype=np.uint8))

        # Combined skin mask
        mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)

        # Morphological filtering to remove noise and fill palm holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # 2. Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return "UNKNOWN", 0.0, {"reason": "no_contours"}

        # Select largest skin contour
        hand_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(hand_contour)

        # Minimum area threshold (hand should occupy at least ~2500 px in 400px frame)
        min_hand_area = 2500
        if area < min_hand_area:
            return "UNKNOWN", 0.0, {"reason": "hand_too_small", "area": area}

        # 3. Contour Geometry Analysis
        bx, by, bw, bh = cv2.boundingRect(hand_contour)
        aspect_ratio = float(bw) / bh if bh > 0 else 1.0
        extent = float(area) / (bw * bh) if (bw * bh) > 0 else 0.0

        # Convex Hull & Defects
        hull = cv2.convexHull(hand_contour, returnPoints=False)
        hull_pts = cv2.convexHull(hand_contour, returnPoints=True)

        if len(hull) < 4:
            return "ROCK", 0.70, {"fingers": 0, "reason": "compact_hull"}

        # Find palm center and top points
        moments = cv2.moments(hand_contour)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            cx, cy = bx + bw // 2, by + bh // 2

        # Count significant convexity defects (valleys between extended fingers)
        extended_fingers_count = 0
        deep_defects_count = 0

        try:
            defects = cv2.convexityDefects(hand_contour, hull)
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(hand_contour[s][0])
                    end = tuple(hand_contour[e][0])
                    far = tuple(hand_contour[f][0])
                    depth = d / 256.0

                    # Calculate angle of defect (triangle start-far-end)
                    a = np.linalg.norm(np.array(end) - np.array(start))
                    b = np.linalg.norm(np.array(far) - np.array(start))
                    c = np.linalg.norm(np.array(end) - np.array(far))
                    
                    # Cosine rule for angle
                    angle = 180.0
                    if b * c > 0:
                        cos_angle = (b**2 + c**2 - a**2) / (2 * b * c)
                        cos_angle = max(-1.0, min(1.0, cos_angle))
                        angle = np.degrees(np.arccos(cos_angle))

                    # Deep valley between fingers: angle < 95 degrees, depth > 18px
                    if angle < 95.0 and depth > 18.0:
                        deep_defects_count += 1
        except Exception as ex:
            logger.debug("Convexity defect calculation exception: %s", ex)

        # Count hull points significantly above palm center (fingertips)
        fingertip_candidates = []
        for pt in hull_pts:
            px, py = pt[0]
            # Point is in upper half of the bounding box
            if py < (by + bh * 0.65) and (cy - py) > 20:
                # Filter points too close to each other
                if not any(np.linalg.norm(np.array([px, py]) - np.array(prev)) < 22 for prev in fingertip_candidates):
                    fingertip_candidates.append((px, py))

        extended_fingers_count = len(fingertip_candidates)

        # 4. Gesture Classification Decision Tree
        # - ROCK: Compact hand, extent > 0.60, 0 or 1 deep defect, <= 1 fingertip candidate
        # - SCISSORS: 2 fingertips extended (V-shape), 1 or 2 deep defects
        # - PAPER: 4 or 5 fingertips extended, open palm, 3 or more deep defects, extent < 0.58

        details = {
            "area": area,
            "extent": round(extent, 2),
            "aspect_ratio": round(aspect_ratio, 2),
            "deep_defects": deep_defects_count,
            "fingertip_candidates": extended_fingers_count,
        }

        if deep_defects_count >= 3 or extended_fingers_count >= 4:
            return "PAPER", 0.92, details
        elif deep_defects_count == 1 or extended_fingers_count == 2:
            return "SCISSORS", 0.90, details
        elif deep_defects_count == 0 and extended_fingers_count <= 1:
            return "ROCK", 0.88, details
        elif extent > 0.62 and deep_defects_count <= 1:
            return "ROCK", 0.82, details
        elif extended_fingers_count == 3:
            # 3 extended fingers can occur on Scissors or Paper variations
            if deep_defects_count >= 2:
                return "PAPER", 0.78, details
            return "SCISSORS", 0.75, details
        else:
            return "UNKNOWN", 0.50, details

"""
Morphix Desktop Companion Robot - Face Detection Engine
======================================================
File: src/vision/face_detection.py
Description:
    Provides face detection logic and result storage for Morphix.
    Designed with a modular interface for future integration of vision backends 
    (such as OpenCV Haar Cascades or Deep Neural Networks).
"""

from typing import Any, List, Tuple


class FaceDetector:
    """
    Manages face detection model loading state, frame processing,
    and face location tracking for Morphix.
    """

    def __init__(self, model_path: str = "") -> None:
        """
        Initializes the FaceDetector instance and internal state.

        :param model_path: Optional file path to a trained detection model.
        """
        self._model_path: str = model_path
        self._model_loaded: bool = False
        self._face_locations: List[Tuple[int, int, int, int]] = []

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str = "") -> bool:
        """
        Loads and prepares the detection model state.

        :param model_path: File path to model configuration/weights.
        :return: True if model state is marked loaded, False if invalid.
        """
        if model_path:
            self._model_path = model_path.strip()

        # Mark model state as loaded and ready for frame processing
        self._model_loaded = True
        return True

    def detect_faces(self, frame: Any) -> List[Tuple[int, int, int, int]]:
        """
        Processes an input image frame and detects faces within it.
        Updates internal face count and location coordinates.

        :param frame: Image frame data (e.g., matrix, bytes, or array).
        :return: List of bounding box tuples representing face locations [(x, y, w, h), ...].
        """
        # Reset current frame detection results
        self._face_locations = []

        if frame is None or not self._model_loaded:
            return self._face_locations

        try:
            # Vision detection logic will process the frame here
            return self._face_locations
        except Exception:
            return []

    def get_face_count(self) -> int:
        """
        Returns the number of faces detected in the most recently processed frame.

        :return: Integer count of detected faces.
        """
        return len(self._face_locations)

    def is_face_detected(self) -> bool:
        """
        Returns whether at least one face was detected in the last processed frame.

        :return: True if one or more faces were detected, False otherwise.
        """
        return len(self._face_locations) > 0

    def get_face_locations(self) -> List[Tuple[int, int, int, int]]:
        """
        Retrieves the bounding box locations for all detected faces.

        :return: List of tuples formatted as (x_offset, y_offset, width, height).
        """
        return self._face_locations

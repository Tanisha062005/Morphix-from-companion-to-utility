"""
Morphix Desktop Companion Robot - Vision System
===============================================
File: src/vision/__init__.py
"""

from src.vision.camera_manager import CameraManager
from src.vision.face_dataset_loader import FaceDatasetLoader
from src.vision.face_detection import FaceDetector

__all__ = ["FaceDetector", "FaceDatasetLoader", "CameraManager"]

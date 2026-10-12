"""
Morphix Desktop Companion Robot - Dynamic Face Dataset Loader & Recognizer
==========================================================================
File: src/vision/face_dataset_loader.py
Description:
    Provides dynamic scanning, preprocessing, feature extraction, and real-time
    LBPH face identification with zero hardcoded identities.
    All identities are derived dynamically from subdirectories in data/known_faces/.
"""

import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Import profile manager helper functions for seamless config integration
from src.config.profile_manager import (
    add_household_member,
    get_household_members,
    get_primary_user,
    load_user_profile,
    remove_household_member,
    save_user_profile,
    set_primary_user,
)

logger = logging.getLogger(__name__)

# Valid image file extensions for dataset scanning
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# Default standard ROI dimensions for LBPH training & inference
DEFAULT_TARGET_SIZE: Tuple[int, int] = (150, 150)

# Default LBPH distance threshold (lower distance = higher similarity)
# Tuned to 85.0 to reliably handle room lighting variations, with a soft match zone up to 90.0
DEFAULT_DISTANCE_THRESHOLD: float = 85.0
DEFAULT_SOFT_DISTANCE_THRESHOLD: float = 90.0

# Default minimum confidence percentage required for positive identification
DEFAULT_MIN_CONFIDENCE: float = 5.0


def find_project_root() -> str:
    """
    Locates the Morphix project root directory based on current file location.

    :return: Absolute path to project root.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # src/vision -> src -> root
    src_dir = os.path.dirname(current_dir)
    root_dir = os.path.dirname(src_dir)
    return root_dir


class FaceDatasetLoader:
    """
    Dynamically scans dataset directories, extracts face features,
    trains an embedded-optimized LBPH face recognizer, and matches faces
    strictly against registered identities.
    """

    def __init__(
        self,
        dataset_dir: Optional[str] = None,
        target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
        distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
        soft_distance_threshold: float = DEFAULT_SOFT_DISTANCE_THRESHOLD,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        auto_load: bool = True,
        async_load: bool = True,
    ) -> None:
        """
        Initializes the dynamic FaceDatasetLoader.

        :param dataset_dir: Custom path to known faces dataset. If None, defaults to data/known_faces/.
        :param target_size: (width, height) tuple for standardizing face ROIs.
        :param distance_threshold: Primary LBPH distance threshold for positive match (default: 85.0).
        :param soft_distance_threshold: Extended soft-match threshold for room lighting variations (default: 90.0).
        :param min_confidence: Minimum confidence percentage (0-100) for a positive match.
        :param auto_load: If True, scans directory and trains recognizer on initialization.
        :param async_load: If True (default), trains recognizer asynchronously in background thread.
        """
        self.project_root: str = find_project_root()
        if dataset_dir:
            self.dataset_dir: str = os.path.abspath(dataset_dir)
        else:
            self.dataset_dir = os.path.join(self.project_root, "data", "known_faces")

        self.target_size: Tuple[int, int] = target_size
        self.distance_threshold: float = distance_threshold
        self.soft_distance_threshold: float = max(soft_distance_threshold, distance_threshold)
        self.min_confidence: float = min_confidence

        # Dynamic label mappings: label_id (int) <-> identity_name (str)
        self.label_to_name: Dict[int, str] = {}
        self.name_to_label: Dict[str, int] = {}
        self.dataset_stats: Dict[str, Any] = {
            "total_identities": 0,
            "total_images": 0,
            "identities": {},
        }

        # Initialize LBPH Recognizer with soft threshold boundary so candidate predictions up to 90.0 are evaluated
        self.recognizer: cv2.face.LBPHFaceRecognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1,
            neighbors=8,
            grid_x=8,
            grid_y=8,
            threshold=self.soft_distance_threshold,
        )
        self._is_trained: bool = False

        # Load Haar Cascade detector for preprocessing uncropped photos
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade: Optional[cv2.CascadeClassifier] = None
        if os.path.exists(cascade_path):
            self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # Ensure directory structure exists
        self.ensure_dataset_structure()

        if auto_load:
            if async_load:
                self.load_and_train_async()
            else:
                self.load_and_train()

    def load_and_train_async(self) -> threading.Thread:
        """
        Scans dataset and trains LBPH face recognizer in a background thread
        to prevent blocking main application startup.
        """
        thread = threading.Thread(target=self.load_and_train, daemon=True, name="FaceLoaderThread")
        thread.start()
        return thread

    def ensure_dataset_structure(self) -> None:
        """
        Ensures the data/known_faces directory exists.
        If empty, creates a generic template directory data/known_faces/User/ with a README.
        """
        os.makedirs(self.dataset_dir, exist_ok=True)

        # Check if any subdirectories exist
        existing_subdirs = [
            d for d in os.listdir(self.dataset_dir)
            if os.path.isdir(os.path.join(self.dataset_dir, d))
            and not d.startswith((".", "_"))
        ]

        if not existing_subdirs:
            # Create generic template folder for user
            user_template_dir = os.path.join(self.dataset_dir, "User")
            os.makedirs(user_template_dir, exist_ok=True)

            readme_path = os.path.join(user_template_dir, "README.txt")
            if not os.path.exists(readme_path):
                readme_content = (
                    "Morphix Known Faces Dataset Guide\n"
                    "===================================\n\n"
                    "To register identities for Morphix Face Recognition:\n"
                    "1. Create a subfolder inside 'data/known_faces/' named after the person (e.g., 'Alice', 'User', 'FamilyMember').\n"
                    "2. Add 3 to 10 clear, well-lit frontal face photos (.jpg, .png, .jpeg, .webp) inside that folder.\n"
                    "3. Morphix will automatically scan, preprocess, and train on all folders dynamically on startup.\n"
                )
                try:
                    with open(readme_path, "w", encoding="utf-8") as f:
                        f.write(readme_content)
                    logger.info(f"Created generic template directory with README at: {user_template_dir}")
                except Exception as e:
                    logger.warning(f"Could not create template README: {e}")

    def scan_dataset(self) -> Dict[str, List[str]]:
        """
        Dynamically scans all subdirectories inside dataset_dir.
        Extracts identity strictly from folder names.

        :return: Dict mapping identity_name -> list of valid image file paths.
        """
        self.ensure_dataset_structure()
        scanned_data: Dict[str, List[str]] = {}

        if not os.path.exists(self.dataset_dir):
            return scanned_data

        for entry in os.scandir(self.dataset_dir):
            if entry.is_dir() and not entry.name.startswith((".", "_")):
                identity_name = entry.name.strip()
                if not identity_name:
                    continue

                image_files: List[str] = []
                for file_entry in os.scandir(entry.path):
                    if file_entry.is_file():
                        _, ext = os.path.splitext(file_entry.name.lower())
                        if ext in VALID_IMAGE_EXTENSIONS:
                            image_files.append(os.path.abspath(file_entry.path))

                scanned_data[identity_name] = sorted(image_files)

        return scanned_data

    def preprocess_face(self, image_input: Any) -> Optional[np.ndarray]:
        """
        Standardizes face image/ROI for LBPH feature extraction & prediction:
        1. Enforces conversion to single-channel 8-bit grayscale (uint8).
        2. Detects face ROI via Haar Cascade if given an uncropped photo.
        3. Resizes consistently to standard target dimensions (150x150 px).
        4. Applies Histogram Equalization (cv2.equalizeHist) on every ROI before training and live prediction.

        :param image_input: Image file path (str) or raw numpy image array.
        :return: Preprocessed 2D uint8 numpy array of shape target_size, or None if invalid.
        """
        if image_input is None:
            return None

        # Load image if file path is provided
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                return None
            try:
                # Read using cv2.imdecode to safely handle unicode/special paths on Windows
                with open(image_input, "rb") as f:
                    file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
            except Exception as e:
                logger.warning(f"Failed to read image '{image_input}': {e}")
                return None
        elif isinstance(image_input, np.ndarray):
            image = image_input.copy()
        else:
            return None

        if image is None or image.size == 0:
            return None

        # 1. Enforce conversion to single-channel Grayscale
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            elif image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 1:
                gray = image[:, :, 0]
            else:
                return None
        elif len(image.shape) == 2:
            gray = image
        else:
            return None

        # Enforce 8-bit uint8 single-channel
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)

        gray = np.ascontiguousarray(gray)

        # 2. If image is significantly larger than target, attempt Haar face detection
        h, w = gray.shape[:2]
        if self.face_cascade is not None and (w > self.target_size[0] * 1.5 or h > self.target_size[1] * 1.5):
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(self.target_size[0] // 2, self.target_size[1] // 2),
            )
            if len(faces) > 0:
                # Use largest detected face
                largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
                fx, fy, fw, fh = largest_face
                gray = gray[fy : fy + fh, fx : fx + fw]

        if gray is None or gray.size == 0:
            return None

        # 3. Resize consistently to standard target dimensions (150x150)
        if gray.shape[0] != self.target_size[1] or gray.shape[1] != self.target_size[0]:
            interp = cv2.INTER_AREA if (gray.shape[0] > self.target_size[1] or gray.shape[1] > self.target_size[0]) else cv2.INTER_CUBIC
            gray = cv2.resize(gray, self.target_size, interpolation=interp)

        # 4. Histogram equalization on every ROI before training and live prediction
        equalized = cv2.equalizeHist(gray)

        return equalized

    def load_and_train(self) -> bool:
        """
        Scans dataset directory, extracts preprocessed face ROIs, assigns dynamic
        integer labels to identities, and trains the OpenCV LBPH face recognizer.

        :return: True if model trained successfully with at least 1 image, False otherwise.
        """
        scanned_dataset = self.scan_dataset()

        self.label_to_name.clear()
        self.name_to_label.clear()
        self._is_trained = False

        faces_list: List[np.ndarray] = []
        labels_list: List[int] = []

        total_images_processed = 0
        identity_image_counts: Dict[str, int] = {}

        current_label_id = 1

        for identity_name, image_paths in scanned_dataset.items():
            valid_images_for_person = 0

            for img_path in image_paths:
                face_roi = self.preprocess_face(img_path)
                if face_roi is not None:
                    faces_list.append(face_roi)
                    labels_list.append(current_label_id)
                    valid_images_for_person += 1
                    total_images_processed += 1

            if valid_images_for_person > 0:
                self.label_to_name[current_label_id] = identity_name
                self.name_to_label[identity_name] = current_label_id
                identity_image_counts[identity_name] = valid_images_for_person
                current_label_id += 1

        self.dataset_stats = {
            "total_identities": len(self.label_to_name),
            "total_images": total_images_processed,
            "identities": identity_image_counts,
        }

        if len(faces_list) == 0:
            logger.info("No face images found in dataset. Recognizer is ready in zero-identity standby mode.")
            return False

        try:
            labels_array = np.array(labels_list, dtype=np.int32)
            self.recognizer.train(faces_list, labels_array)
            self._is_trained = True
            logger.info(
                f"Successfully trained LBPH Face Recognizer with {len(faces_list)} images across "
                f"{len(self.label_to_name)} identities: {list(self.label_to_name.values())}"
            )
            return True
        except Exception as e:
            logger.error(f"Error training LBPH face recognizer: {e}")
            self._is_trained = False
            return False

    def identify_face(self, face_roi: Any) -> Tuple[bool, str, float]:
        """
        Identifies a face ROI against the dynamically trained dataset.

        :param face_roi: Grayscale or BGR cropped face image or frame.
        :return: Tuple of (is_known: bool, matched_name: str, confidence: float).
                 Returns (False, "unknown", 0.0) if no match or untrained.
        """
        if not self._is_trained or face_roi is None:
            return (False, "unknown", 0.0)

        # Standardize input face ROI with 150x150 grayscale histogram equalization
        processed_roi = self.preprocess_face(face_roi)
        if processed_roi is None or processed_roi.size == 0:
            return (False, "unknown", 0.0)

        try:
            label_id, distance = self.recognizer.predict(processed_roi)

            # In LBPH:
            # - label_id is -1 if no trained face falls within the distance threshold.
            # - distance is a Chi-square difference. Lower distance = higher similarity.
            if (
                label_id == -1
                or label_id not in self.label_to_name
                or distance > self.soft_distance_threshold
                or distance >= 1e10
            ):
                return (False, "unknown", 0.0)

            matched_name = self.label_to_name[label_id]

            # Calculate normalized confidence percentage (0.0 to 100.0%)
            # Reliable match zone: distance <= distance_threshold (<= 85.0)
            # Soft match zone: 85.0 < distance <= soft_distance_threshold (<= 90.0) for normal lighting variations
            if distance <= self.distance_threshold:
                confidence = max(
                    15.0,
                    min(100.0, round((1.0 - (distance / 100.0)) * 100.0, 2)),
                )
            else:
                confidence = max(
                    5.0,
                    min(100.0, round((1.0 - (distance / 100.0)) * 100.0, 2)),
                )

            # Enforce minimum positive confidence requirement
            if confidence < self.min_confidence:
                return (False, "unknown", 0.0)

            return (True, matched_name, float(confidence))

        except Exception as e:
            logger.error(f"Face identification error: {e}")
            return (False, "unknown", 0.0)

    def is_trained(self) -> bool:
        """
        Returns True if recognizer is currently trained on at least one face.
        """
        return self._is_trained

    def get_known_identities(self) -> List[str]:
        """
        Returns a list of all dynamically registered identity names.
        """
        return list(self.label_to_name.values())

    def get_dataset_stats(self) -> Dict[str, Any]:
        """
        Returns summary statistics of loaded dataset.
        """
        return self.dataset_stats.copy()

    def save_model(self, model_path: str) -> bool:
        """
        Saves trained LBPH model weights to an XML/YAML file.

        :param model_path: Target file path to write model weights.
        :return: True if saved successfully, False otherwise.
        """
        if not self._is_trained:
            logger.warning("Cannot save untrained model.")
            return False

        try:
            os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
            self.recognizer.save(model_path)
            return True
        except Exception as e:
            logger.error(f"Failed to save model to '{model_path}': {e}")
            return False

    def load_model(self, model_path: str, label_map: Dict[int, str]) -> bool:
        """
        Loads pre-trained LBPH model weights and restores label mappings.

        :param model_path: Path to saved model weights.
        :param label_map: Dictionary mapping integer label IDs to identity names.
        :return: True if loaded successfully, False otherwise.
        """
        if not os.path.exists(model_path):
            return False

        try:
            self.recognizer.read(model_path)
            self.label_to_name = label_map.copy()
            self.name_to_label = {v: k for k, v in label_map.items()}
            self._is_trained = True
            return True
        except Exception as e:
            logger.error(f"Failed to load model from '{model_path}': {e}")
            return False


# Module-level convenience functions
def get_user_profile_primary_user() -> str:
    """
    Helper function to get primary user name from config.
    """
    return get_primary_user()


def set_user_profile_primary_user(name: str) -> bool:
    """
    Helper function to set primary user name in config.
    """
    return set_primary_user(name)

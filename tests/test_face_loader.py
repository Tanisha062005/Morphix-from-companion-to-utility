"""
Morphix Desktop Companion Robot - Face Dataset Loader Test Suite
================================================================
File: tests/test_face_loader.py
Description:
    Unit tests for dynamic user profile management, face dataset scanning,
    LBPH feature extraction, dataset training, and identification matching.
"""

import os
import shutil
import tempfile
import unittest
import cv2
import numpy as np

from src.config.profile_manager import (
    add_household_member,
    get_household_members,
    get_primary_user,
    load_user_profile,
    remove_household_member,
    save_user_profile,
    set_primary_user,
)
from src.vision.face_dataset_loader import FaceDatasetLoader


def create_synthetic_face(seed: int, width: int = 150, height: int = 150) -> np.ndarray:
    """Generates deterministic synthetic face-like image for testing."""
    img = np.full((height, width), fill_value=120, dtype=np.uint8)
    np.random.seed(seed)
    noise = np.random.randint(-15, 15, (height, width), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    eye_offset = (seed * 7) % 20
    cv2.circle(img, (40 + eye_offset, 50), 12, 40, -1)
    cv2.circle(img, (110 - eye_offset, 50), 12, 40, -1)
    cv2.line(img, (75, 65), (75, 95), 50, 3)
    cv2.ellipse(img, (75, 115), (25 + (seed % 10), 10), 0, 0, 180, 40, 3)
    return img


class TestFaceLoader(unittest.TestCase):
    """Unit tests for FaceDatasetLoader and ProfileManager."""

    def test_generic_user_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_config = os.path.join(temp_dir, "test_profile.json")

            profile = load_user_profile(config_path=temp_config)
            self.assertEqual(profile.get("primary_user"), "User")
            self.assertIsInstance(profile.get("household_members"), list)

            success = set_primary_user("Alex", config_path=temp_config)
            self.assertTrue(success)
            self.assertEqual(get_primary_user(config_path=temp_config), "Alex")

            add_household_member("MemberOne", config_path=temp_config)
            add_household_member("MemberTwo", config_path=temp_config)
            members = get_household_members(config_path=temp_config)
            self.assertIn("MemberOne", members)
            self.assertIn("MemberTwo", members)

            remove_household_member("MemberOne", config_path=temp_config)
            members_after = get_household_members(config_path=temp_config)
            self.assertNotIn("MemberOne", members_after)

    def test_face_dataset_scanner_and_training(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = os.path.join(temp_dir, "known_faces")

            loader = FaceDatasetLoader(dataset_dir=dataset_dir, auto_load=False)
            self.assertTrue(os.path.exists(dataset_dir))

            person_a_dir = os.path.join(dataset_dir, "Alice")
            person_b_dir = os.path.join(dataset_dir, "Bob")
            os.makedirs(person_a_dir, exist_ok=True)
            os.makedirs(person_b_dir, exist_ok=True)

            for i in range(3):
                img_a = create_synthetic_face(seed=10 + i)
                cv2.imwrite(os.path.join(person_a_dir, f"face_{i}.png"), img_a)
                img_b = create_synthetic_face(seed=50 + i)
                cv2.imwrite(os.path.join(person_b_dir, f"face_{i}.png"), img_b)

            success = loader.load_and_train()
            self.assertTrue(success)
            self.assertTrue(loader.is_trained())

            test_img_a = create_synthetic_face(seed=10)
            is_known_a, matched_name_a, conf_a = loader.identify_face(test_img_a)
            self.assertTrue(is_known_a)
            self.assertEqual(matched_name_a, "Alice")

            # Unknown non-matching image pattern
            unknown_img = np.zeros((150, 150), dtype=np.uint8)
            unknown_img[20:130, 20:130] = 255
            is_known_u, matched_name_u, _ = loader.identify_face(unknown_img)
            self.assertFalse(is_known_u)
            self.assertEqual(matched_name_u, "unknown")

    def test_preprocess_face_enforcement(self) -> None:
        """Tests that preprocess_face enforces 8-bit grayscale, 150x150 resizing, and histogram equalization."""
        loader = FaceDatasetLoader(auto_load=False)

        # 1. 3-channel BGR image with non-standard dimensions
        bgr_img = np.full((240, 320, 3), 100, dtype=np.uint8)
        processed = loader.preprocess_face(bgr_img)
        self.assertIsNotNone(processed)
        self.assertEqual(processed.shape, (150, 150))
        self.assertEqual(processed.dtype, np.uint8)
        self.assertEqual(len(processed.shape), 2)

        # 2. 4-channel BGRA image
        bgra_img = np.full((100, 100, 4), 150, dtype=np.uint8)
        processed_bgra = loader.preprocess_face(bgra_img)
        self.assertIsNotNone(processed_bgra)
        self.assertEqual(processed_bgra.shape, (150, 150))
        self.assertEqual(processed_bgra.dtype, np.uint8)

        # 3. None / empty input
        self.assertIsNone(loader.preprocess_face(None))
        self.assertIsNone(loader.preprocess_face(np.array([])))

    def test_distance_threshold_and_soft_match_zone(self) -> None:
        """Tests reliable match (<= 85.0), soft match zone (85.0 - 90.0), and rejection (> 90.0)."""
        from unittest.mock import MagicMock

        loader = FaceDatasetLoader(auto_load=False)
        loader._is_trained = True
        loader.label_to_name = {1: "Tanisha"}
        loader.name_to_label = {"Tanisha": 1}

        mock_rec = MagicMock()
        loader.recognizer = mock_rec

        sample_face = np.full((150, 150), 128, dtype=np.uint8)

        # 1. Distance = 75.0 (Reliable match)
        mock_rec.predict.return_value = (1, 75.0)
        is_known, name, conf = loader.identify_face(sample_face)
        self.assertTrue(is_known)
        self.assertEqual(name, "Tanisha")
        self.assertGreaterEqual(conf, 15.0)

        # 2. Distance = 88.0 (Soft match zone between 85.0 and 90.0)
        mock_rec.predict.return_value = (1, 88.0)
        is_known_soft, name_soft, conf_soft = loader.identify_face(sample_face)
        self.assertTrue(is_known_soft)
        self.assertEqual(name_soft, "Tanisha")
        self.assertGreaterEqual(conf_soft, 5.0)

        # 3. Distance = 95.0 (Beyond soft match zone -> Rejected)
        mock_rec.predict.return_value = (1, 95.0)
        is_known_rej, name_rej, conf_rej = loader.identify_face(sample_face)
        self.assertFalse(is_known_rej)
        self.assertEqual(name_rej, "unknown")
        self.assertEqual(conf_rej, 0.0)


if __name__ == "__main__":
    unittest.main()

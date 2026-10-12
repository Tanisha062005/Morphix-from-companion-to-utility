"""
Morphix Desktop Companion Robot - Voice Profile & Silent Surveillance Test Suite
================================================================================
File: tests/test_surveillance_and_profile.py
Description:
    Comprehensive unit and integration tests for Step 2 of Morphix:
    1. Voice Profile Intent matching & service execution.
    2. Voice Face Enrollment & dynamic retraining.
    3. Silent Surveillance Intent matching & service controls.
    4. CameraManager motion filter, intruder capture, and cooldown logic.
"""

import json
import os
import shutil
import sys
import tempfile
import time
import unittest
import cv2
import numpy as np

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config.profile_manager import get_primary_user, load_user_profile, set_primary_user
from src.intent.intent_classifier import IntentClassifier, IntentType
from src.intent.profile_plugin import ProfileIntentPlugin, ProfileService
from src.intent.surveillance_plugin import SurveillanceIntentPlugin, SurveillanceService
from src.vision.camera_manager import CameraManager
from src.vision.face_dataset_loader import FaceDatasetLoader


class TestProfilePlugin(unittest.TestCase):
    """Tests voice profile setup and face registration regex matching."""

    def setUp(self):
        self.plugin = ProfileIntentPlugin()
        self.classifier = IntentClassifier()

    def test_set_name_english(self):
        queries = [
            ("My name is Alex", "Alex"),
            ("Call me Jordan", "Jordan"),
            ("Set my name to Morgan", "Morgan"),
            ("Change my name to Sam", "Sam"),
            ("I am Taylor", "Taylor"),
        ]
        for query, expected_name in queries:
            res = self.plugin.match(query)
            self.assertIsNotNone(res, f"Failed matching: {query}")
            self.assertEqual(res.intent, IntentType.SET_USER_NAME)
            self.assertEqual(res.parameters.get("name"), expected_name)
            self.assertFalse(res.parameters.get("is_hindi"))

    def test_set_name_hindi(self):
        queries = [
            ("Mera naam Rohan hai", "Rohan"),
            ("Mujhe Priya bulao", "Priya"),
            ("Mera naam Amit rakho", "Amit"),
        ]
        for query, expected_name in queries:
            res = self.plugin.match(query)
            self.assertIsNotNone(res, f"Failed matching: {query}")
            self.assertEqual(res.intent, IntentType.SET_USER_NAME)
            self.assertEqual(res.parameters.get("name"), expected_name)
            self.assertTrue(res.parameters.get("is_hindi"))

    def test_get_name_queries(self):
        res_en = self.plugin.match("Who am I")
        self.assertEqual(res_en.intent, IntentType.GET_USER_NAME)

        res_hi = self.plugin.match("Mera naam kya hai")
        self.assertEqual(res_hi.intent, IntentType.GET_USER_NAME)
        self.assertTrue(res_hi.parameters.get("is_hindi"))

    def test_enroll_face_queries(self):
        en_queries = [
            "Register my face",
            "Enroll my face",
            "Capture my face",
            "Scan my face",
            "Take my face photos",
        ]
        for q in en_queries:
            res = self.plugin.match(q)
            self.assertIsNotNone(res, f"Failed matching: {q}")
            self.assertEqual(res.intent, IntentType.ENROLL_FACE)
            self.assertFalse(res.parameters.get("is_hindi"))

        hi_queries = [
            "Mera face scan karo",
            "Mera chehra scan karo",
            "Face register karo",
            "Mera face register kar do",
        ]
        for q in hi_queries:
            res = self.plugin.match(q)
            self.assertIsNotNone(res, f"Failed matching: {q}")
            self.assertEqual(res.intent, IntentType.ENROLL_FACE)
            self.assertTrue(res.parameters.get("is_hindi"))


class TestSurveillancePlugin(unittest.TestCase):
    """Tests surveillance control regex matching and service responses."""

    def setUp(self):
        self.plugin = SurveillanceIntentPlugin()

    def test_start_surveillance_matching(self):
        queries = [
            ("Activate surveillance mode", False),
            ("Start guard mode", False),
            ("Watch my room", False),
            ("Guard my room", False),
            ("Surveillance chalu karo", True),
            ("Guard mode start karo", True),
            ("Room ki surveillance karo", True),
        ]
        for q, is_hi in queries:
            res = self.plugin.match(q)
            self.assertIsNotNone(res, f"Failed matching: {q}")
            self.assertEqual(res.intent, IntentType.START_SURVEILLANCE)
            self.assertEqual(res.parameters.get("is_hindi"), is_hi)

    def test_stop_surveillance_matching(self):
        queries = [
            ("Stop surveillance", False),
            ("I am back", False),
            ("Deactivate surveillance mode", False),
            ("Guard mode band karo", True),
            ("Main wapas aa gaya", True),
            ("Surveillance off karo", True),
        ]
        for q, is_hi in queries:
            res = self.plugin.match(q)
            self.assertIsNotNone(res, f"Failed matching: {q}")
            self.assertEqual(res.intent, IntentType.STOP_SURVEILLANCE)
            self.assertEqual(res.parameters.get("is_hindi"), is_hi)

    def test_surveillance_status_matching(self):
        queries = [
            ("Surveillance status", False),
            ("Did anyone enter my room", False),
            ("Any intruders detected", False),
            ("Kya koi aaya tha", True),
            ("Surveillance report kya hai", True),
        ]
        for q, is_hi in queries:
            res = self.plugin.match(q)
            self.assertIsNotNone(res, f"Failed matching: {q}")
            self.assertEqual(res.intent, IntentType.GET_SURVEILLANCE_STATUS)
            self.assertEqual(res.parameters.get("is_hindi"), is_hi)

    def test_delete_surveillance_captures_matching(self):
        queries = [
            ("delete surveillance images", False),
            ("clear captured photos", False),
            ("delete intruder photos", False),
            ("delete photos", False),
            ("clear captures", False),
            ("delete images", False),
            ("clear surveillance images", False),
            ("delete surveillance captures", False),
            ("surveillance photos delete karo", True),
            ("surveillance captures delete karo", True),
            ("photos delete karo", True),
            ("saari surveillance photos hata do", True),
        ]
        for q, is_hi in queries:
            res = self.plugin.match(q)
            self.assertIsNotNone(res, f"Failed matching: {q}")
            self.assertEqual(res.intent, IntentType.DELETE_SURVEILLANCE_CAPTURES)
            self.assertEqual(res.parameters.get("is_hindi"), is_hi)


class TestCameraManagerAndSurveillance(unittest.TestCase):
    """Tests CameraManager motion filtering, intruder capture, and face enrollment."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.captures_dir = os.path.join(self.temp_dir, "captures")
        self.dataset_dir = os.path.join(self.temp_dir, "known_faces")
        os.makedirs(self.captures_dir, exist_ok=True)
        os.makedirs(self.dataset_dir, exist_ok=True)

        # Create known user dataset
        user_dir = os.path.join(self.dataset_dir, "RegisteredUser")
        os.makedirs(user_dir, exist_ok=True)

        # Create synthetic face for RegisteredUser
        for i in range(3):
            img = np.full((150, 150), fill_value=120, dtype=np.uint8)
            cv2.circle(img, (45, 50), 10, 30, -1)
            cv2.circle(img, (105, 50), 10, 30, -1)
            cv2.line(img, (75, 60), (75, 90), 40, 3)
            cv2.ellipse(img, (75, 115), (30, 10), 0, 0, 180, 30, 3)
            cv2.imwrite(os.path.join(user_dir, f"sample_{i}.png"), img)

        self.face_loader = FaceDatasetLoader(dataset_dir=self.dataset_dir, auto_load=True)
        self.camera_manager = CameraManager(
            camera_index=99,  # Unused hardware index to force software mode
            captures_dir=self.captures_dir,
            face_loader=self.face_loader,
            cooldown_seconds=1.0,
        )

    def tearDown(self):
        if self.camera_manager.is_surveillance_active():
            self.camera_manager.stop_surveillance()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_motion_filter_logic(self):
        """Tests that identical frames produce NO motion and differing frames trigger motion."""
        frame1 = np.full((480, 640, 3), 100, dtype=np.uint8)
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        blur1 = cv2.GaussianBlur(gray1, (21, 21), 0)

        # Identical frame -> No motion
        motion, area = self.camera_manager._detect_motion(blur1, blur1)
        self.assertFalse(motion)
        self.assertEqual(area, 0)

        # Differing frame (large movement) -> Motion detected
        frame2 = frame1.copy()
        cv2.rectangle(frame2, (100, 100), (300, 300), (255, 255, 255), -1)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        blur2 = cv2.GaussianBlur(gray2, (21, 21), 0)

        motion, area = self.camera_manager._detect_motion(blur1, blur2)
        self.assertTrue(motion)
        self.assertGreater(area, 1500)

    def test_silent_intruder_capture_and_logging(self):
        """Tests silent snapshot saving and structured json logging for unknown intruder face."""
        # Initial frame (background)
        frame_bg = np.full((480, 640, 3), 100, dtype=np.uint8)
        _, prev_gray, _ = self.camera_manager.process_surveillance_frame(frame_bg, None)

        # Unknown intruder frame (motion alert + face ROI)
        frame_intruder = frame_bg.copy()
        cv2.rectangle(frame_intruder, (50, 50), (400, 400), (255, 255, 255), -1)

        # Mock face_cascade to detect an unrecognized face ROI
        from unittest.mock import MagicMock
        mock_cascade = MagicMock()
        mock_cascade.detectMultiScale.return_value = np.array([[100, 100, 150, 150]])
        self.camera_manager.face_cascade = mock_cascade

        motion, _, result = self.camera_manager.process_surveillance_frame(frame_intruder, prev_gray)
        self.assertTrue(motion)
        self.assertEqual(result["action"], "INTRUDER_DETECTED")
        self.assertEqual(self.camera_manager.get_session_intruder_count(), 1)

        # Verify event log json file exists and has 1 record
        log_path = os.path.join(self.captures_dir, "surveillance_events.json")
        self.assertTrue(os.path.exists(log_path))
        with open(log_path, "r", encoding="utf-8") as f:
            events = json.load(f)
            self.assertEqual(len(events), 1)
            self.assertIn("EVT_", events[0]["event_id"])

        # Test cooldown: immediate second frame should NOT increment count
        _, _, result2 = self.camera_manager.process_surveillance_frame(frame_intruder, prev_gray)
        self.assertEqual(self.camera_manager.get_session_intruder_count(), 1)

    def test_known_user_and_motion_without_face_bypassed(self):
        """Tests that pure scene motion without face and known user faces are bypassed without saving snapshots."""
        frame_bg = np.full((480, 640, 3), 100, dtype=np.uint8)
        _, prev_gray, _ = self.camera_manager.process_surveillance_frame(frame_bg, None)

        # 1. Motion without a face ROI -> MOTION_NO_FACE, is_authorized=False, no capture
        frame_motion = frame_bg.copy()
        cv2.rectangle(frame_motion, (50, 50), (400, 400), (255, 255, 255), -1)

        from unittest.mock import MagicMock
        mock_cascade = MagicMock()
        mock_cascade.detectMultiScale.return_value = ()
        self.camera_manager.face_cascade = mock_cascade

        motion, _, result = self.camera_manager.process_surveillance_frame(frame_motion, prev_gray)
        self.assertTrue(motion)
        self.assertEqual(result["action"], "MOTION_NO_FACE")
        self.assertFalse(result["face_detected"])
        self.assertFalse(result["is_authorized"])
        self.assertEqual(self.camera_manager.get_session_intruder_count(), 0)

        # 2. Motion with known user face ROI -> AUTHORIZED_PASS, is_authorized=True, no capture
        mock_loader = MagicMock()
        mock_loader.identify_face.return_value = (True, "RegisteredUser", 95.0)
        self.camera_manager.face_loader = mock_loader

        mock_cascade.detectMultiScale.return_value = np.array([[100, 100, 150, 150]])
        motion, _, result_known = self.camera_manager.process_surveillance_frame(frame_motion, prev_gray)
        self.assertTrue(motion)
        self.assertEqual(result_known["action"], "AUTHORIZED_PASS")
        self.assertTrue(result_known["is_authorized"])
        self.assertTrue(result_known["face_detected"])
        self.assertEqual(self.camera_manager.get_session_intruder_count(), 0)

        # 3. Motion with authorized name variation ("tanisha") -> AUTHORIZED_PASS
        mock_loader.identify_face.return_value = (False, "tanisha", 88.0)
        motion, _, result_tanisha = self.camera_manager.process_surveillance_frame(frame_motion, prev_gray)
        self.assertTrue(motion)
        self.assertEqual(result_tanisha["action"], "AUTHORIZED_PASS")
        self.assertTrue(result_tanisha["is_authorized"])
        self.assertEqual(self.camera_manager.get_session_intruder_count(), 0)

    def test_face_enrollment_pipeline_and_synthetic_rejection(self):
        """Tests capture_face_samples rejects synthetic fallback when camera is unavailable,
        and cleanly captures 6 normalized face crops when valid frames are provided."""
        from unittest.mock import MagicMock

        # 1. Camera unavailable -> Rejects synthetic fallback, returns False and 0 files
        self.camera_manager._cap = None
        success_offline, saved_offline, msg_offline = self.camera_manager.capture_face_samples("OfflineUser", count=6)
        self.assertFalse(success_offline)
        self.assertEqual(len(saved_offline), 0)
        self.assertIn("unavailable", msg_offline.lower())
        self.assertFalse(os.path.exists(os.path.join(self.dataset_dir, "OfflineUser")))

        # 2. Camera active with valid human face ROIs -> Captures 6 normalized samples & retrains
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        self.camera_manager._cap = mock_cap

        # Create a sample frame containing a face-like pattern
        mock_frame = np.full((480, 640, 3), fill_value=120, dtype=np.uint8)
        cv2.circle(mock_frame, (320, 240), 80, (200, 200, 200), -1)
        cv2.circle(mock_frame, (290, 210), 12, (50, 50, 50), -1)
        cv2.circle(mock_frame, (350, 210), 12, (50, 50, 50), -1)
        cv2.line(mock_frame, (320, 230), (320, 260), (40, 40, 40), 3)

        self.camera_manager.read_frame = MagicMock(return_value=mock_frame)

        mock_cascade = MagicMock()
        mock_cascade.detectMultiScale.return_value = np.array([[200, 120, 240, 240]])
        self.camera_manager.face_cascade = mock_cascade

        new_person = "NewResident"
        success, saved_files, msg = self.camera_manager.capture_face_samples(new_person, count=6)

        self.assertTrue(success)
        self.assertEqual(len(saved_files), 6)
        for f in saved_files:
            self.assertTrue(os.path.exists(f))
            saved_img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            self.assertEqual(saved_img.shape, (150, 150))
            self.assertEqual(saved_img.dtype, np.uint8)

        # Verify FaceDatasetLoader dynamically and instantly retrained in-memory with the new person
        known_identities = self.face_loader.get_known_identities()
        self.assertIn(new_person, known_identities)
        self.assertTrue(self.face_loader.is_trained())

    def test_whitelisting_flexible_profile_matching(self):
        """Tests that normalized and flexible matching correctly identifies authorized users."""
        # 1. Direct known match
        self.assertTrue(self.camera_manager._is_identity_authorized("RegisteredUser", is_known=True))
        self.assertTrue(self.camera_manager._is_identity_authorized("registereduser", is_known=True))

        # 2. Match against primary user / aliases / household members
        self.assertTrue(self.camera_manager._is_identity_authorized("tanisha", is_known=False))
        self.assertTrue(self.camera_manager._is_identity_authorized("Tanisha", is_known=False))
        self.assertTrue(self.camera_manager._is_identity_authorized("User", is_known=False))
        self.assertTrue(self.camera_manager._is_identity_authorized("user", is_known=False))

        # 3. Unknown identities
        self.assertFalse(self.camera_manager._is_identity_authorized("unknown", is_known=False))
        self.assertFalse(self.camera_manager._is_identity_authorized("IntruderStranger", is_known=False))
        self.assertFalse(self.camera_manager._is_identity_authorized("", is_known=False))

    def test_clear_surveillance_captures_and_event_logs(self):
        """Tests that clear_surveillance_captures deletes image files and clears events log."""
        # 1. Create sample intruder photos in captures_dir
        for i in range(4):
            sample_file = os.path.join(self.captures_dir, f"intruder_{i}.jpg")
            with open(sample_file, "wb") as f:
                f.write(b"\xFF\xD8\xFF\xE0\x00\x10JFIF")

        # 2. Write event records to surveillance_events.json
        events_file = os.path.join(self.captures_dir, "surveillance_events.json")
        with open(events_file, "w", encoding="utf-8") as f:
            json.dump([{"event_id": "EVT_1"}, {"event_id": "EVT_2"}], f)

        self.camera_manager._session_intruder_count = 2

        # 3. Purge captures
        success, deleted_count, msg = self.camera_manager.clear_surveillance_captures()
        self.assertTrue(success)
        self.assertEqual(deleted_count, 4)

        # 4. Verify image files deleted
        remaining_images = [
            f for f in os.listdir(self.captures_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
        self.assertEqual(len(remaining_images), 0)

        # 5. Verify events log reset
        with open(events_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertEqual(data, [])

        self.assertEqual(self.camera_manager.get_session_intruder_count(), 0)

    def test_delete_surveillance_captures_service_response(self):
        """Tests SurveillanceService.delete_surveillance_captures voice responses."""
        from unittest.mock import patch

        service = SurveillanceService.get_instance()

        # Mock camera clear returning count > 0
        with patch.object(CameraManager, "get_instance") as mock_cm:
            mock_cam = mock_cm.return_value
            mock_cam.clear_surveillance_captures.return_value = (True, 3, "Deleted 3 capture image(s).")

            resp_en = service.delete_surveillance_captures(is_hindi=False)
            self.assertIn("deleted successfully", resp_en)

            resp_hi = service.delete_surveillance_captures(is_hindi=True)
            self.assertIn("3 photos delete", resp_hi)

            # Mock 0 captures
            mock_cam.clear_surveillance_captures.return_value = (True, 0, "Deleted 0 capture image(s).")
            resp_en_zero = service.delete_surveillance_captures(is_hindi=False)
            self.assertIn("No surveillance captures found", resp_en_zero)

    def test_calculate_blur_score(self):
        """Tests calculate_blur_score returns higher variance for sharp images and lower for blurred images."""
        # 1. Uniform/flat image -> score 0.0
        flat = np.full((100, 100), fill_value=128, dtype=np.uint8)
        self.assertEqual(self.camera_manager.calculate_blur_score(flat), 0.0)

        # 2. Sharp high-contrast checkerboard/edge pattern -> high score
        sharp = np.zeros((100, 100), dtype=np.uint8)
        sharp[::2, ::2] = 255
        sharp_score = self.camera_manager.calculate_blur_score(sharp)
        self.assertGreater(sharp_score, 1000.0)

        # 3. Heavily blurred version -> significantly lower score
        blurred = cv2.GaussianBlur(sharp, (25, 25), 0)
        blurred_score = self.camera_manager.calculate_blur_score(blurred)
        self.assertLess(blurred_score, sharp_score)

        # 4. None / invalid input -> 0.0
        self.assertEqual(self.camera_manager.calculate_blur_score(None), 0.0)

    def test_surveillance_blur_filtering_and_fallback(self):
        """Tests that blurry frames are rejected and 3-second continuous intruder presence allows fallback."""
        from unittest.mock import MagicMock

        frame_bg = np.full((480, 640, 3), 100, dtype=np.uint8)
        _, prev_gray, _ = self.camera_manager.process_surveillance_frame(frame_bg, None)

        # Create motion frame with a heavily blurred face ROI
        frame_blurry = frame_bg.copy()
        cv2.rectangle(frame_blurry, (50, 50), (400, 400), (255, 255, 255), -1)
        # Apply heavy blur to the whole frame so both face_roi and frame have low variance
        frame_blurry = cv2.GaussianBlur(frame_blurry, (51, 51), 0)

        mock_cascade = MagicMock()
        mock_cascade.detectMultiScale.return_value = np.array([[100, 100, 150, 150]])
        self.camera_manager.face_cascade = mock_cascade

        # Mock face identifier returning unknown face
        mock_loader = MagicMock()
        mock_loader.identify_face.return_value = (False, "unknown", 0.0)
        self.camera_manager.face_loader = mock_loader

        # 1. Blurry frame on first detection -> rejected due to motion blur, count stays 0
        self.camera_manager.blur_threshold = 80.0
        motion, _, result = self.camera_manager.process_surveillance_frame(frame_blurry, prev_gray)
        self.assertTrue(motion)
        self.assertEqual(result["action"], "INTRUDER_DETECTED")
        self.assertLess(result["blur_score"], 80.0)
        self.assertEqual(self.camera_manager.get_session_intruder_count(), 0)

        # 2. Simulate 3.5 seconds passing with continuous intruder presence -> Fallback saves snapshot
        self.camera_manager._intruder_first_seen_time = time.time() - 3.5
        _, _, result_fallback = self.camera_manager.process_surveillance_frame(frame_blurry, prev_gray)
        self.assertEqual(self.camera_manager.get_session_intruder_count(), 1)


if __name__ == "__main__":
    unittest.main()


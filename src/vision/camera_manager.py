"""
Morphix Desktop Companion Robot - Real-time Silent Surveillance & Camera Manager
================================================================================
File: src/vision/camera_manager.py
Description:
    Provides embedded-optimized background camera thread (640x480 @ 15 FPS)
    featuring non-blocking DirectShow capture, motion filtering, face
    identification, silent intruder snapshot capture with 10s cooldown,
    structured event logging, and voice face enrollment.
"""

import concurrent.futures
import datetime
import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.config.profile_manager import get_household_members, get_primary_user
from src.vision.face_dataset_loader import FaceDatasetLoader, find_project_root

logger = logging.getLogger("Morphix.CameraManager")


class CameraManager:
    """
    Manages non-blocking video capture, low-CPU motion detection,
    facial identification, intruder capture, and voice face enrollment.
    """

    _instance: Optional["CameraManager"] = None
    _lock = threading.RLock()

    BLUR_THRESHOLD: float = 80.0

    @classmethod
    def get_instance(cls, **kwargs) -> "CameraManager":
        """
        Singleton access pattern for shared camera management across plugins.
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(**kwargs)
            return cls._instance

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 15,
        cooldown_seconds: float = 10.0,
        blur_threshold: float = 80.0,
        captures_dir: Optional[str] = None,
        face_loader: Optional[FaceDatasetLoader] = None,
    ) -> None:
        """
        Initializes the CameraManager.

        :param camera_index: Hardware camera device index.
        :param width: Frame capture width (default: 640).
        :param height: Frame capture height (default: 480).
        :param fps: Target capture frame rate (default: 15).
        :param cooldown_seconds: Minimum cooldown between intruder captures (default: 10.0s).
        :param blur_threshold: Laplacian variance threshold for sharpness filtering (default: 80.0).
        :param captures_dir: Directory for storing intruder snapshots and event logs.
        :param face_loader: FaceDatasetLoader instance for face identification.
        """
        self.project_root: str = find_project_root()
        self.camera_index: int = camera_index
        self.target_width: int = width
        self.target_height: int = height
        self.target_fps: int = fps
        self.cooldown_seconds: float = cooldown_seconds
        self.blur_threshold: float = blur_threshold

        if captures_dir:
            self.captures_dir: str = os.path.abspath(captures_dir)
        else:
            self.captures_dir = os.path.join(self.project_root, "data", "surveillance_captures")

        os.makedirs(self.captures_dir, exist_ok=True)
        self.events_log_path: str = os.path.join(self.captures_dir, "surveillance_events.json")

        # Face dataset loader for face matching
        self.face_loader: FaceDatasetLoader = face_loader or FaceDatasetLoader(auto_load=True)

        # Threading state
        self._is_surveillance_active: bool = False
        self._surveillance_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Motion detection, blur filter, & capture state
        self._last_capture_time: float = 0.0
        self._session_start_time: float = 0.0
        self._session_intruder_count: int = 0
        self._total_frames_processed: int = 0
        self._intruder_first_seen_time: float = 0.0

        # VideoCapture handle with Reentrant Lock to prevent self-deadlocks
        self._cap: Optional[cv2.VideoCapture] = None
        self._cap_lock = threading.RLock()

        # Haar cascade detector for ROI extraction
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade: Optional[cv2.CascadeClassifier] = None
        if os.path.exists(cascade_path):
            self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # Eye display callback hook
        self._display_state_callback = None

    def set_display_callback(self, callback) -> None:
        """
        Registers a callback to change eye animation state (e.g., SENTRY_SCAN).
        """
        self._display_state_callback = callback

    def _trigger_display_state(self, state_name: str) -> None:
        """
        Invokes display state callback if registered.
        """
        if self._display_state_callback and callable(self._display_state_callback):
            try:
                self._display_state_callback(state_name)
            except Exception as e:
                logger.warning(f"Error calling display callback: {e}")

    def _open_camera(self, timeout_sec: float = 3.0) -> bool:
        """
        Opens hardware camera device with DirectShow (CAP_DSHOW) on Windows
        and a non-blocking timeout to prevent thread hangs.

        :param timeout_sec: Max seconds to wait for camera hardware initialization.
        :return: True if camera is opened and ready, False otherwise.
        """
        with self._cap_lock:
            if self._cap is not None and self._cap.isOpened():
                return True

            def _init_cap() -> Optional[cv2.VideoCapture]:
                try:
                    # Windows DirectShow backend to prevent MSMF hang
                    cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                    if not cap.isOpened():
                        cap = cv2.VideoCapture(self.camera_index)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
                        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                        return cap
                    return None
                except Exception as ex:
                    logger.debug(f"Exception during VideoCapture init: {ex}")
                    return None

            try:
                # Use a short-lived executor with timeout to prevent hardware driver hangs
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_init_cap)
                    cap = future.result(timeout=timeout_sec)

                if cap is not None and cap.isOpened():
                    self._cap = cap
                    logger.info(
                        f"Camera [{self.camera_index}] opened successfully "
                        f"({self.target_width}x{self.target_height} @ {self.target_fps}fps via DSHOW)."
                    )
                    return True
                else:
                    logger.warning(f"Camera [{self.camera_index}] not detected. Running in software mode.")
                    self._cap = None
                    return False

            except concurrent.futures.TimeoutError:
                logger.warning(f"Camera [{self.camera_index}] open timed out ({timeout_sec}s). Running in software mode.")
                self._cap = None
                return False
            except Exception as e:
                logger.error(f"Error opening camera: {e}")
                self._cap = None
                return False

    def _release_camera(self) -> None:
        """
        Releases the camera device safely.
        """
        with self._cap_lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception as e:
                    logger.warning(f"Error releasing camera: {e}")
                self._cap = None
                logger.info("Camera released.")

    def read_frame(self) -> Optional[np.ndarray]:
        """
        Safely captures a single frame from the camera.
        """
        with self._cap_lock:
            if self._cap is None or not self._cap.isOpened():
                return None
            try:
                ret, frame = self._cap.read()
                if ret and frame is not None and frame.size > 0:
                    return frame
                return None
            except Exception as e:
                logger.warning(f"Error reading frame from camera: {e}")
                return None

    def is_surveillance_active(self) -> bool:
        """
        Returns True if surveillance guard mode is currently running.
        """
        return self._is_surveillance_active

    def get_session_intruder_count(self) -> int:
        """
        Returns count of intruder events captured during the active/latest session.
        """
        return self._session_intruder_count

    def start_surveillance(self) -> bool:
        """
        Activates real-time background surveillance mode.

        :return: True if surveillance started successfully.
        """
        if self._is_surveillance_active:
            logger.info("Surveillance mode is already active.")
            return True

        self._session_start_time = time.time()
        self._session_intruder_count = 0
        self._total_frames_processed = 0
        self._stop_event.clear()

        # Open camera safely
        self._open_camera(timeout_sec=3.0)

        # Set Sentry Eye Animation
        self._trigger_display_state("SENTRY_SCAN")

        self._is_surveillance_active = True
        self._surveillance_thread = threading.Thread(
            target=self._surveillance_loop,
            name="Morphix-SurveillanceThread",
            daemon=True,
        )
        self._surveillance_thread.start()
        logger.info("Silent Surveillance guard mode activated.")
        return True

    def stop_surveillance(self) -> Dict[str, Any]:
        """
        Deactivates surveillance mode, restores display, and returns session report.

        :return: Dict containing session summary statistics.
        """
        if not self._is_surveillance_active:
            return {
                "status": "INACTIVE",
                "duration_seconds": 0.0,
                "intruders_captured": 0,
                "frames_processed": 0,
            }

        self._stop_event.set()
        self._is_surveillance_active = False

        if self._surveillance_thread and self._surveillance_thread.is_alive():
            self._surveillance_thread.join(timeout=2.0)

        self._release_camera()

        # Restore normal eyes
        self._trigger_display_state("IDLE_NEUTRAL")

        session_duration = round(time.time() - self._session_start_time, 2)
        summary = {
            "status": "STOPPED",
            "duration_seconds": session_duration,
            "intruders_captured": self._session_intruder_count,
            "frames_processed": self._total_frames_processed,
        }
        logger.info(f"Surveillance stopped. Session summary: {summary}")
        return summary

    def _detect_motion(
        self, prev_gray: Optional[np.ndarray], curr_gray: np.ndarray, min_area: int = 1500
    ) -> Tuple[bool, int]:
        """
        Efficient motion filter using Gaussian blur and absolute frame difference.

        :param prev_gray: Previous smoothed grayscale frame.
        :param curr_gray: Current smoothed grayscale frame.
        :param min_area: Minimum motion contour pixel area to qualify as motion.
        :return: Tuple of (motion_detected: bool, total_motion_area: int).
        """
        if prev_gray is None or curr_gray is None:
            return False, 0

        # Frame difference
        frame_diff = cv2.absdiff(prev_gray, curr_gray)
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_area = 0

        for c in contours:
            area = cv2.contourArea(c)
            if area >= min_area:
                total_area += int(area)

        return (total_area > 0), total_area

    def calculate_blur_score(self, image_gray: Any) -> float:
        """
        Returns the Laplacian variance score of the image/face crop.
        Higher score indicates sharper edges and focus; lower score indicates motion blur or defocus.

        :param image_gray: Grayscale or BGR image numpy array.
        :return: Float variance of the Laplacian.
        """
        if image_gray is None:
            return 0.0
        try:
            if isinstance(image_gray, np.ndarray):
                if image_gray.size == 0:
                    return 0.0
                if len(image_gray.shape) == 3:
                    image_gray = cv2.cvtColor(image_gray, cv2.COLOR_BGR2GRAY)
                return float(cv2.Laplacian(image_gray, cv2.CV_64F).var())
            return 0.0
        except Exception as e:
            logger.debug(f"Error computing Laplacian blur score: {e}")
            return 0.0

    def _log_surveillance_event(
        self,
        event_type: str,
        image_path: str,
        confidence: float,
        note: str = "",
    ) -> Dict[str, Any]:
        """
        Appends a structured event entry to surveillance_events.json atomically.

        :param event_type: Event classification ("UNKNOWN", "MOTION_ALERT", "KNOWN_USER").
        :param image_path: Path to captured snapshot.
        :param confidence: Face recognition confidence score.
        :param note: Optional descriptive notes.
        :return: Created event record dict.
        """
        now = datetime.datetime.now()
        event_id = f"EVT_{now.strftime('%Y%m%d_%H%M%S')}_{int(now.microsecond / 1000):03d}"

        event_record = {
            "event_id": event_id,
            "timestamp": now.isoformat(),
            "event_type": event_type,
            "image_path": os.path.relpath(image_path, self.project_root).replace("\\", "/"),
            "confidence": round(float(confidence), 2),
            "note": note,
        }

        events_list: List[Dict[str, Any]] = []
        if os.path.exists(self.events_log_path):
            try:
                with open(self.events_log_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        events_list = data
            except Exception as e:
                logger.warning(f"Could not read existing events log: {e}")

        events_list.append(event_record)

        try:
            with open(self.events_log_path, "w", encoding="utf-8") as f:
                json.dump(events_list, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to write surveillance event log: {e}")

        return event_record

    def _is_identity_authorized(self, matched_name: str, is_known: bool) -> bool:
        """
        Determines whether a recognized identity corresponds to an authorized user/household member.
        Normalizes all names (.strip().lower()) and matches flexibly against folder names, profile name, and aliases.

        :param matched_name: Identity name returned by FaceDatasetLoader.
        :param is_known: Boolean flag indicating if LBPH recognized the face against known dataset.
        :return: True if authorized, False otherwise.
        """
        if not matched_name or not isinstance(matched_name, str):
            return False

        norm_name = matched_name.strip().lower()
        if norm_name in ("unknown", "", "none"):
            return False

        # If LBPH positively matched a face from data/known_faces, it is a registered identity
        if is_known:
            return True

        # Retrieve profile primary user, household members, and registered folder names
        primary_user = get_primary_user().strip().lower()
        household = [m.strip().lower() for m in get_household_members()]
        dataset_identities = [k.strip().lower() for k in self.face_loader.get_known_identities()]

        # Common authorized alias set (including default user names, profile names, and household members)
        authorized_names = {
            primary_user,
            "user",
            "tanisha",
            "owner",
            "admin",
            "master",
        } | set(household) | set(dataset_identities)

        # Remove empty strings
        authorized_names = {n for n in authorized_names if n}

        # 1. Exact normalized match
        if norm_name in authorized_names:
            return True

        # 2. Flexible substring / prefix / suffix match (e.g. "tanisha" matching "tanisha_profile" or "user" matching "registereduser")
        for auth in authorized_names:
            if auth in norm_name or norm_name in auth:
                return True

        return False

    def process_surveillance_frame(
        self, frame: np.ndarray, prev_gray: Optional[np.ndarray]
    ) -> Tuple[bool, Optional[np.ndarray], Dict[str, Any]]:
        """
        Processes a single video frame for motion, face detection, and silent intruder capture.

        :param frame: Raw BGR input frame.
        :param prev_gray: Previous smoothed grayscale frame.
        :return: Tuple of (motion_detected, current_smoothed_gray, result_dict).
        """
        self._total_frames_processed += 1
        result = {
            "action": "NONE",
            "motion_detected": False,
            "motion_area": 0,
            "face_detected": False,
            "is_known": False,
            "is_authorized": False,
            "name": "unknown",
            "confidence": 0.0,
        }

        if frame is None or frame.size == 0:
            return False, prev_gray, result

        # Convert to Grayscale & apply Gaussian blur for motion filter
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_gray = cv2.GaussianBlur(gray, (21, 21), 0)

        motion_detected, motion_area = self._detect_motion(prev_gray, blurred_gray)
        result["motion_detected"] = motion_detected
        result["motion_area"] = motion_area

        if not motion_detected:
            return False, blurred_gray, result

        # When motion is detected, check for human face ROIs
        faces = []
        if self.face_cascade is not None:
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(60, 60),
            )

        now_time = time.time()
        cooldown_elapsed = (now_time - self._last_capture_time) >= self.cooldown_seconds

        if len(faces) > 0:
            result["face_detected"] = True
            # Extract primary face ROI
            fx, fy, fw, fh = max(faces, key=lambda rect: rect[2] * rect[3])
            face_roi = gray[fy : fy + fh, fx : fx + fw]

            is_known, matched_name, confidence = self.face_loader.identify_face(face_roi)
            is_authorized = self._is_identity_authorized(matched_name, is_known)

            result["is_known"] = is_known
            result["is_authorized"] = is_authorized
            result["name"] = matched_name
            result["confidence"] = confidence

            if is_authorized:
                # Authorized user -> Return is_authorized=True and action AUTHORIZED_PASS
                # Completely bypass snapshot capture and event logging
                result["action"] = "AUTHORIZED_PASS"
                self._intruder_first_seen_time = 0.0
                logger.debug(f"Recognized authorized user '{matched_name}' ({confidence:.1f}%). Bypassing surveillance capture.")
            else:
                # UNKNOWN intruder or unrecognized face ROI!
                result["action"] = "INTRUDER_DETECTED"

                if self._intruder_first_seen_time == 0.0:
                    self._intruder_first_seen_time = now_time

                # Calculate sharpness via Laplacian variance (check face ROI and frame)
                roi_blur = self.calculate_blur_score(face_roi)
                frame_blur = self.calculate_blur_score(gray)
                blur_score = max(roi_blur, frame_blur)
                result["blur_score"] = round(blur_score, 2)

                time_in_view = now_time - self._intruder_first_seen_time
                is_sharp_enough = blur_score >= self.blur_threshold
                is_fallback_timeout = time_in_view >= 3.0  # Fallback if continuously present for 3s in dim lighting

                if cooldown_elapsed:
                    if is_sharp_enough or is_fallback_timeout:
                        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        img_filename = f"intruder_{timestamp_str}.jpg"
                        img_path = os.path.join(self.captures_dir, img_filename)

                        # Save silent snapshot
                        cv2.imwrite(img_path, frame)
                        self._last_capture_time = now_time
                        self._session_intruder_count += 1
                        self._intruder_first_seen_time = 0.0

                        # Log structured event
                        note_suffix = f"(Blur Score: {blur_score:.2f})" if is_sharp_enough else f"(Fallback Timeout: {time_in_view:.1f}s, Blur Score: {blur_score:.2f})"
                        self._log_surveillance_event(
                            event_type="UNKNOWN",
                            image_path=img_path,
                            confidence=confidence,
                            note=f"Unrecognized face detected during active guard mode {note_suffix}",
                        )
                        logger.info(
                            f"[CameraManager]: Sharp intruder snapshot captured: Saved '{img_filename}' "
                            f"(Blur Score: {blur_score:.2f} >= {self.blur_threshold}, Total session: {self._session_intruder_count})."
                        )
                    else:
                        logger.debug(
                            f"[CameraManager]: Frame rejected due to motion blur "
                            f"(Blur Score: {blur_score:.2f} < {self.blur_threshold}, Time in view: {time_in_view:.1f}s)."
                        )
        else:
            # Non-human / scene motion without verified face contour (camera shake / background movement) -> Ignore capture
            result["face_detected"] = False
            result["is_authorized"] = False
            result["action"] = "MOTION_NO_FACE"
            self._intruder_first_seen_time = 0.0
            logger.debug(f"Scene motion detected (area: {motion_area} px) without verified face contour. Bypassing capture.")

        return motion_detected, blurred_gray, result

    def _surveillance_loop(self) -> None:
        """
        Background worker thread executing real-time surveillance loop.
        """
        prev_gray: Optional[np.ndarray] = None
        frame_interval = 1.0 / self.target_fps

        logger.info("Surveillance background loop started.")

        while not self._stop_event.is_set():
            loop_start = time.time()

            frame = self.read_frame()
            if frame is not None:
                _, prev_gray, _ = self.process_surveillance_frame(frame, prev_gray)
            else:
                # Sleep briefly if camera has no frame
                time.sleep(0.05)

            # Maintain target FPS
            elapsed = time.time() - loop_start
            sleep_time = max(0.005, frame_interval - elapsed)
            time.sleep(sleep_time)

        logger.info("Surveillance background loop terminated.")

    def capture_face_samples(
        self, person_name: str, count: int = 6, max_attempts: int = 40
    ) -> Tuple[bool, List[str], str]:
        """
        Multi-Frame Robust Voice-driven Face Enrollment:
        1. Captures 6-8 distinct face frames with 280ms delay for slight pose/expression variation.
        2. Enforces strict Haar Cascade verification: saves ONLY valid detected face ROIs (no uncropped full-frames).
        3. Normalizes and equalizes each cropped ROI (150x150 px) via preprocess_face before saving.
        4. Completely rejects synthetic drawings / gray squares when camera is unavailable.
        5. Triggers instant in-memory model retraining via FaceDatasetLoader.load_and_train().

        :param person_name: Identity name for directory.
        :param count: Target number of distinct face samples to capture (default: 6).
        :param max_attempts: Maximum camera frame reads before timing out.
        :return: Tuple of (success: bool, list_of_saved_paths: List[str], message: str).
        """
        if not person_name or not isinstance(person_name, str) or not person_name.strip():
            return False, [], "Invalid person name for face registration."

        cleaned_name = person_name.strip()
        target_person_dir = os.path.join(self.face_loader.dataset_dir, cleaned_name)

        was_surveillance_active = self._is_surveillance_active
        if was_surveillance_active:
            self.stop_surveillance()

        # Open camera safely if not already opened
        cam_opened_locally = False
        if self._cap is None or not self._cap.isOpened():
            cam_opened_locally = self._open_camera(timeout_sec=3.0)

        # Reject synthetic fallbacks if camera is unavailable/disconnected
        if self._cap is None or not self._cap.isOpened():
            logger.warning("Face enrollment aborted: Camera hardware is not available. Rejecting synthetic fallback.")
            if was_surveillance_active:
                self.start_surveillance()
            return False, [], "Camera is disconnected or unavailable. Real camera capture required for face enrollment."

        saved_files: List[str] = []
        captured_count = 0
        attempts = 0

        try:
            os.makedirs(target_person_dir, exist_ok=True)

            while captured_count < count and attempts < max_attempts:
                attempts += 1
                # 250ms - 300ms capture delay between frames to ensure slight pose/expression variation
                time.sleep(0.28)

                frame = self.read_frame()
                if frame is None or frame.size == 0:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Strict Haar Cascade verification: only save if a valid face ROI is actively detected
                if self.face_cascade is None:
                    continue

                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(60, 60),
                )

                if len(faces) == 0:
                    logger.debug(f"Enrollment attempt {attempts}: No face detected. Skipping frame.")
                    continue

                # Crop primary detected face ROI with slight margin
                fx, fy, fw, fh = max(faces, key=lambda rect: rect[2] * rect[3])
                pad = int(fw * 0.1)
                y1 = max(0, fy - pad)
                y2 = min(frame.shape[0], fy + fh + pad)
                x1 = max(0, fx - pad)
                x2 = min(frame.shape[1], fx + fw + pad)
                face_crop = frame[y1:y2, x1:x2]

                # Normalize & histogram equalize ROI consistently to 150x150 px via preprocess_face
                preprocessed_roi = self.face_loader.preprocess_face(face_crop)
                if preprocessed_roi is None or preprocessed_roi.size == 0:
                    continue

                img_path = os.path.join(target_person_dir, f"face_{captured_count + 1}.jpg")
                success = cv2.imwrite(img_path, preprocessed_roi)
                if success:
                    saved_files.append(img_path)
                    captured_count += 1
                    logger.info(f"Captured enrollment sample {captured_count}/{count} for '{cleaned_name}'.")

        finally:
            if cam_opened_locally:
                self._release_camera()

            if was_surveillance_active:
                self.start_surveillance()

        if captured_count > 0:
            # Trigger instant in-memory model retraining to update the active LBPH model immediately
            retrain_success = self.face_loader.load_and_train()
            logger.info(
                f"Face enrollment completed for '{cleaned_name}': {captured_count} real face samples saved. "
                f"Active model retrained in-memory: {retrain_success}."
            )
            return True, saved_files, f"Successfully captured {captured_count} face sample(s) for '{cleaned_name}'."
        else:
            # Clean up empty directory if no valid samples were collected
            try:
                if os.path.exists(target_person_dir) and not os.listdir(target_person_dir):
                    os.rmdir(target_person_dir)
            except Exception:
                pass
            return False, [], "No valid face detected during capture session. Please face the camera directly."

    def get_recent_events(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves the latest surveillance events recorded in surveillance_events.json.

        :param limit: Maximum number of events to return.
        :return: List of event records (most recent first).
        """
        if not os.path.exists(self.events_log_path):
            return []

        try:
            with open(self.events_log_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data[-limit:][::-1]
        except Exception as e:
            logger.warning(f"Error reading surveillance events log: {e}")

        return []

    def clear_surveillance_captures(self) -> Tuple[bool, int, str]:
        """
        Voice-activated Surveillance Snapshot Deletion & Event Log Purge:
        Safely deletes all snapshot image files (.jpg, .jpeg, .png, .webp) from
        data/surveillance_captures/ and resets surveillance_events.json to an empty list.

        :return: Tuple of (success: bool, deleted_count: int, message: str).
        """
        deleted_count = 0
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        if os.path.exists(self.captures_dir):
            try:
                for filename in os.listdir(self.captures_dir):
                    file_path = os.path.join(self.captures_dir, filename)
                    if os.path.isfile(file_path):
                        _, ext = os.path.splitext(filename.lower())
                        if ext in image_extensions:
                            try:
                                os.remove(file_path)
                                deleted_count += 1
                            except Exception as ex:
                                logger.warning(f"Could not remove capture file '{file_path}': {ex}")
            except Exception as e:
                logger.error(f"Error scanning captures directory '{self.captures_dir}': {e}")

        # Reset surveillance events log atomically to an empty list
        try:
            os.makedirs(os.path.dirname(self.events_log_path), exist_ok=True)
            with open(self.events_log_path, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error clearing surveillance event log: {e}")

        # Reset active session intruder counter
        self._session_intruder_count = 0
        logger.info(f"Cleared {deleted_count} surveillance capture image(s) and reset event logs.")

        return True, deleted_count, f"Deleted {deleted_count} capture image(s) and reset event logs."

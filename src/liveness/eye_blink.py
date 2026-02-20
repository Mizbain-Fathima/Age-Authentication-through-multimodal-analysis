"""
Eye Blink Liveness Detection

Detects natural eye blinks using MediaPipe Tasks FaceLandmarker.
Uses Eye Aspect Ratio (EAR) to detect open → closed → open transitions.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional
from collections import deque
from loguru import logger
import threading
from pathlib import Path

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.vision import face_landmarker
from mediapipe import Image as MPImage, ImageFormat


class EyeBlinkDetector:
    """
    Eye blink detection using MediaPipe Tasks FaceLandmarker
    
    Detects natural eye blinks by tracking Eye Aspect Ratio (EAR)
    across video frames. A blink is detected as an open → closed → open transition.
    
    Thread-safe implementation using thread-local storage for MediaPipe.
    """
    
    # Thread-local storage for MediaPipe instances
    _local = threading.local()
    
    # Eye landmark indices (MediaPipe 468 landmarks)
    # Standard MediaPipe eye landmark indices
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    
    # EAR thresholds (calibrated for MediaPipe Tasks FaceLandmarker)
    EAR_CLOSED = 0.22  # Eye closed threshold
    EAR_OPEN = 0.26   # Eye open threshold
    
    def __init__(self):
        """Initialize EyeBlinkDetector with MediaPipe Tasks FaceLandmarker"""
        # MediaPipe Tasks model path - will be created per thread
        model_path = self._get_model_path()
        base_options = base_options_module.BaseOptions(model_asset_path=model_path)
        options = face_landmarker.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.face_landmarker_options = options
        
        logger.info("Initialized EyeBlinkDetector")
    
    def _get_model_path(self) -> str:
        """Get path to face landmarker model"""
        # Resolve project root: src/liveness/eye_blink.py -> project_root
        project_root = Path(__file__).resolve().parents[2]
        model_path = project_root / "models" / "face_landmarker.task"
        
        if model_path.exists():
            return str(model_path)
        
        raise RuntimeError(
            f"FaceLandmarker model not found at: {model_path}. "
            "Please ensure models/face_landmarker.task exists in the project root."
        )
    
    @property
    def face_landmarker(self):
        """Get thread-local MediaPipe FaceLandmarker instance"""
        if not hasattr(self._local, 'face_landmarker'):
            self._local.face_landmarker = face_landmarker.FaceLandmarker.create_from_options(
                self.face_landmarker_options
            )
        return self._local.face_landmarker
    
    def _calculate_ear(self, landmarks, eye_indices: List[int]) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) for a single eye
        
        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        
        Args:
            landmarks: MediaPipe landmarks list (direct from face_landmarks[0])
            eye_indices: List of 6 landmark indices [p1, p2, p3, p4, p5, p6]
        
        Returns:
            EAR value (float)
        """
        if len(eye_indices) < 6 or len(landmarks) < max(eye_indices) + 1:
            return 0.0
        
        try:
            # Extract landmark points (landmarks[i] already has .x, .y, .z)
            p1 = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
            p2 = np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y])
            p3 = np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])
            p4 = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
            p5 = np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y])
            p6 = np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])
            
            # Vertical distances (eye opening)
            v1 = np.linalg.norm(p2 - p6)
            v2 = np.linalg.norm(p3 - p5)
            
            # Horizontal distance (eye width)
            h = np.linalg.norm(p1 - p4)
            
            if h == 0:
                return 0.0
            
            # EAR formula
            ear = (v1 + v2) / (2.0 * h)
            return float(ear)
        except (IndexError, AttributeError):
            return 0.0

    def _detect_blinks_from_precomputed(self, precomputed: List[Dict]) -> Dict:
        """Run blink state machine on precomputed per-frame results (ear_value or landmarks)."""
        if not precomputed:
            return {"blink_count": 0, "blink_score": None}
        prev_eye_state = "open"
        closed_frame_count = 0
        blink_count = 0
        face_detected = False
        ear_window = deque(maxlen=3)
        for result in precomputed:
            if not result.get("face_detected"):
                continue
            face_detected = True
            if "ear_value" in result and result["ear_value"] > 0:
                avg_ear = result["ear_value"]
            elif result.get("landmarks") and len(result["landmarks"]) > max(max(self.LEFT_EYE_INDICES), max(self.RIGHT_EYE_INDICES)):
                landmarks = result["landmarks"]
                left_ear = self._calculate_ear(landmarks, self.LEFT_EYE_INDICES)
                right_ear = self._calculate_ear(landmarks, self.RIGHT_EYE_INDICES)
                if left_ear == 0.0 and right_ear == 0.0:
                    continue
                avg_ear = (left_ear + right_ear) / 2.0
            else:
                continue
            ear_window.append(avg_ear)
            smoothed_ear = min(ear_window) if ear_window else avg_ear
            if smoothed_ear < self.EAR_CLOSED:
                closed_frame_count += 1
                current_state = "closed" if (closed_frame_count >= 2 or (closed_frame_count == 1 and smoothed_ear < 0.20)) else prev_eye_state
            elif smoothed_ear > self.EAR_OPEN:
                current_state = "open"
                closed_frame_count = 0
            else:
                current_state = prev_eye_state
                if smoothed_ear > self.EAR_CLOSED:
                    closed_frame_count = 0
            if prev_eye_state == "closed" and current_state == "open":
                blink_count += 1
            prev_eye_state = current_state
        if not face_detected:
            return {"blink_count": 0, "blink_score": None}
        if blink_count == 0:
            blink_score = 0.0
        elif blink_count == 1:
            blink_score = 0.4
        elif blink_count == 2:
            blink_score = 0.7
        else:
            blink_score = 1.0
        blink_score = max(0.0, min(1.0, blink_score))
        logger.info(f"Blink detection: {blink_count} blinks detected, score: {blink_score:.2f}")
        return {"blink_count": blink_count, "blink_score": float(blink_score)}

    def detect_blinks(
        self,
        frames: Optional[List[np.ndarray]] = None,
        precomputed: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Detect eye blinks in a sequence of video frames.
        If precomputed is provided, use it instead of running MediaPipe.

        Args:
            frames: List of BGR images (numpy arrays); can be None if precomputed provided.
            precomputed: Optional list of per-frame dicts with 'face_detected', 'ear_value' or 'landmarks'.

        Returns:
            {
                "blink_count": int,
                "blink_score": float | None
            }
            
            blink_score rules:
            - 0 blinks → 0.0
            - 1 blink → 0.4
            - 2 blinks → 0.7
            - ≥3 blinks → 1.0
            
            If no face detected → blink_score = None
        """
        if precomputed is not None:
            return self._detect_blinks_from_precomputed(precomputed)
        if not frames or len(frames) == 0:
            return {
                "blink_count": 0,
                "blink_score": None
            }
        # Blink state machine with temporal smoothing and minimum closed frames (anti-noise)
        prev_eye_state = "open"  # "open" or "closed"
        closed_frame_count = 0  # Consecutive frames with closed eyes
        blink_count = 0
        face_detected = False
        
        # Temporal smoothing: rolling window for EAR (captures fast blinks between frames)
        ear_window = deque(maxlen=3)  # 3-frame window
        
        # Process each frame
        for frame in frames:
            try:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = MPImage(image_format=ImageFormat.SRGB, data=rgb_frame)
                
                # Detect face landmarks
                detection_result = self.face_landmarker.detect(mp_image)
                
                # Check if face detected and landmarks available
                if (detection_result.face_landmarks and 
                    len(detection_result.face_landmarks) > 0 and
                    len(detection_result.face_landmarks[0]) > max(max(self.LEFT_EYE_INDICES), max(self.RIGHT_EYE_INDICES))):
                    
                    face_detected = True
                    # MediaPipe Tasks: face_landmarks[0] is already a list of landmarks
                    landmarks = detection_result.face_landmarks[0]
                    
                    # Calculate EAR for left and right eyes
                    left_ear = self._calculate_ear(landmarks, self.LEFT_EYE_INDICES)
                    right_ear = self._calculate_ear(landmarks, self.RIGHT_EYE_INDICES)
                    
                    # Skip if EAR calculation failed
                    if left_ear == 0.0 and right_ear == 0.0:
                        continue
                    
                    # Average both eyes
                    avg_ear = (left_ear + right_ear) / 2.0
                    
                    # Add to temporal smoothing window
                    ear_window.append(avg_ear)
                    
                    # Use MIN EAR within window (captures fast blinks)
                    smoothed_ear = min(ear_window) if len(ear_window) > 0 else avg_ear
                    
                    # Determine current eye state with softened closed frames requirement
                    if smoothed_ear < self.EAR_CLOSED:
                        # Eye appears closed - increment closed frame counter
                        closed_frame_count += 1
                        # Softened requirement: 2 consecutive frames OR 1 frame with strong closure (< 0.20)
                        if closed_frame_count >= 2 or (closed_frame_count == 1 and smoothed_ear < 0.20):
                            current_state = "closed"
                        else:
                            # Not enough consecutive frames - maintain previous state
                            current_state = prev_eye_state
                    elif smoothed_ear > self.EAR_OPEN:
                        # Eye is clearly open
                        current_state = "open"
                        closed_frame_count = 0  # Reset closed frame counter
                    else:
                        # In between thresholds - maintain previous state
                        current_state = prev_eye_state
                        # Reset closed frame counter if we're moving away from closed
                        if smoothed_ear > self.EAR_CLOSED:
                            closed_frame_count = 0
                    
                    # Detect blink: CLOSED → OPEN transition
                    # Increment blink_count ONLY when transitioning from closed to open
                    if prev_eye_state == "closed" and current_state == "open":
                        blink_count += 1
                    
                    # Update state
                    prev_eye_state = current_state
                    
            except (IndexError, AttributeError, ValueError):
                # Skip frame silently (no per-frame logging)
                continue
            except Exception:
                # Skip frame silently (no per-frame logging)
                continue
        
        # Calculate blink_score based on blink_count
        if not face_detected:
            return {
                "blink_count": 0,
                "blink_score": None
            }
        
        # Score calculation based on blink count
        # NOTE: Low scores are BY DESIGN and CORRECT for real users
        # - 0 blinks → 0.0 (no liveness evidence)
        # - 1 blink → 0.4 (40% is CORRECT for a single blink in short recording)
        # - 2 blinks → 0.7 (70% for natural double-blink pattern)
        # - ≥3 blinks → 1.0 (strong liveness evidence)
        # DO NOT inflate these scores - they are honest measurements
        if blink_count == 0:
            blink_score = 0.0
        elif blink_count == 1:
            blink_score = 0.4  # CORRECT: 1 blink = 40% confidence
        elif blink_count == 2:
            blink_score = 0.7  # CORRECT: 2 blinks = 70% confidence
        else:  # blink_count >= 3
            blink_score = 1.0  # CORRECT: 3+ blinks = 100% confidence
        
        # Clamp to [0.0, 1.0]
        blink_score = max(0.0, min(1.0, blink_score))
        
        # Single summary log
        logger.info(f"Blink detection: {blink_count} blinks detected, score: {blink_score:.2f}")
        
        return {
            "blink_count": blink_count,
            "blink_score": float(blink_score)
        }


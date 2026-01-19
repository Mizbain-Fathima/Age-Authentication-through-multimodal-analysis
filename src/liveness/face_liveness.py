"""
Face Liveness Detection
Detects real faces vs photos/videos using multiple techniques:
- Eye blink detection
- Head movement analysis
- Texture analysis (anti-spoofing)
- Temporal consistency
"""
import cv2
import numpy as np
from collections import deque
import mediapipe as mp
from loguru import logger
from typing import List, Dict, Tuple, Optional

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import MIN_FACE_FRAMES, BLINK_THRESHOLD, MOTION_THRESHOLD


class FaceLivenessDetector:
    """
    Comprehensive face liveness detection using multiple signals
    """
    
    def __init__(self, min_frames=MIN_FACE_FRAMES, blink_threshold=BLINK_THRESHOLD,
                 motion_threshold=MOTION_THRESHOLD):
        self.min_frames = min_frames
        self.blink_threshold = blink_threshold
        self.motion_threshold = motion_threshold
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Landmark indices for eye aspect ratio
        # Left eye landmarks
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        # Right eye landmarks
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        # Mouth landmarks for lip detection
        self.MOUTH_OUTER = [61, 291, 0, 17, 269, 405, 314, 17, 84, 181]
        self.MOUTH_INNER = [78, 308, 13, 14, 312, 317, 402, 14, 87, 178]
        
        # History buffers
        self.ear_history = deque(maxlen=30)
        self.pose_history = deque(maxlen=30)
        self.landmark_history = deque(maxlen=30)
        
        # State tracking
        self.blink_counter = 0
        self.frame_count = 0
        self.last_landmarks = None
        
        logger.info("Initialized FaceLivenessDetector")
    
    def _calculate_ear(self, landmarks, eye_indices):
        """
        Calculate Eye Aspect Ratio (EAR)
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        """
        p1 = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
        p2 = np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y])
        p3 = np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])
        p4 = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
        p5 = np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y])
        p6 = np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])
        
        # Vertical distances
        v1 = np.linalg.norm(p2 - p6)
        v2 = np.linalg.norm(p3 - p5)
        
        # Horizontal distance
        h = np.linalg.norm(p1 - p4)
        
        if h == 0:
            return 0.0
        
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def _calculate_mar(self, landmarks):
        """
        Calculate Mouth Aspect Ratio (MAR) for lip movement detection
        """
        # Upper and lower lip points
        upper = np.array([landmarks[13].x, landmarks[13].y])
        lower = np.array([landmarks[14].x, landmarks[14].y])
        left = np.array([landmarks[78].x, landmarks[78].y])
        right = np.array([landmarks[308].x, landmarks[308].y])
        
        vertical = np.linalg.norm(upper - lower)
        horizontal = np.linalg.norm(left - right)
        
        if horizontal == 0:
            return 0.0
        
        mar = vertical / horizontal
        return mar
    
    def _estimate_head_pose(self, landmarks, image_shape):
        """
        Estimate head pose (pitch, yaw, roll) from landmarks
        """
        # 3D model points
        model_points = np.array([
            (0.0, 0.0, 0.0),          # Nose tip
            (0.0, -330.0, -65.0),     # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),   # Right eye right corner
            (-150.0, -150.0, -125.0), # Left mouth corner
            (150.0, -150.0, -125.0)   # Right mouth corner
        ], dtype=np.float64)
        
        # 2D image points from landmarks
        h, w = image_shape[:2]
        image_points = np.array([
            (landmarks[1].x * w, landmarks[1].y * h),    # Nose tip
            (landmarks[152].x * w, landmarks[152].y * h), # Chin
            (landmarks[263].x * w, landmarks[263].y * h), # Left eye
            (landmarks[33].x * w, landmarks[33].y * h),   # Right eye
            (landmarks[287].x * w, landmarks[287].y * h), # Left mouth
            (landmarks[57].x * w, landmarks[57].y * h)    # Right mouth
        ], dtype=np.float64)
        
        # Camera matrix (approximate)
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        dist_coeffs = np.zeros((4, 1))
        
        # Solve PnP
        success, rotation_vec, translation_vec = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs
        )
        
        if success:
            # Convert rotation vector to Euler angles
            rotation_mat, _ = cv2.Rodrigues(rotation_vec)
            pose_mat = cv2.hconcat([rotation_mat, translation_vec])
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
            
            pitch = euler_angles[0][0]
            yaw = euler_angles[1][0]
            roll = euler_angles[2][0]
            
            return pitch, yaw, roll
        
        return None
    
    def _analyze_texture(self, face_region):
        """
        Analyze face texture for anti-spoofing
        Uses Local Binary Pattern (LBP) based features
        """
        if face_region is None or face_region.size == 0:
            return 0.5
        
        # Convert to grayscale
        if len(face_region.shape) == 3:
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_region
        
        # Compute Laplacian variance (blur detection)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # High frequency content analysis
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # Calculate ratio of high to low frequencies
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        
        # Low frequency region (center)
        low_freq = magnitude[center_h-30:center_h+30, center_w-30:center_w+30].sum()
        total = magnitude.sum()
        
        high_freq_ratio = 1 - (low_freq / (total + 1e-8))
        
        # Texture score (higher = more likely real)
        texture_score = min(1.0, (laplacian_var / 500) * 0.5 + high_freq_ratio * 0.5)
        
        return texture_score
    
    def _calculate_motion_score(self):
        """
        Calculate motion consistency score from landmark history
        """
        if len(self.landmark_history) < 3:
            return 0.0
        
        motions = []
        for i in range(1, len(self.landmark_history)):
            prev_landmarks = self.landmark_history[i-1]
            curr_landmarks = self.landmark_history[i]
            
            # Calculate average movement of key points
            total_motion = 0
            for idx in [1, 33, 263, 152]:  # Nose, eyes, chin
                if idx < len(prev_landmarks) and idx < len(curr_landmarks):
                    prev_pt = np.array([prev_landmarks[idx].x, prev_landmarks[idx].y])
                    curr_pt = np.array([curr_landmarks[idx].x, curr_landmarks[idx].y])
                    total_motion += np.linalg.norm(curr_pt - prev_pt)
            
            motions.append(total_motion / 4)
        
        # Real faces have natural micro-movements
        motion_variance = np.var(motions) if motions else 0
        motion_mean = np.mean(motions) if motions else 0
        
        # Score based on natural movement patterns
        # Too still = likely photo, too erratic = suspicious
        if motion_mean < 0.001:  # Very still
            return 0.2
        elif motion_mean > 0.1:  # Too much movement
            return 0.5
        else:
            # Natural movement range
            return min(1.0, 0.6 + motion_variance * 100)
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame and update liveness state
        
        Args:
            frame: BGR image from webcam
        
        Returns:
            Dictionary with detection results
        """
        self.frame_count += 1
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        output = {
            'face_detected': False,
            'blink_detected': False,
            'ear_value': 0.0,
            'mar_value': 0.0,
            'head_pose': None,
            'texture_score': 0.5,
            'motion_score': 0.0,
            'landmarks': None
        }
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            output['face_detected'] = True
            output['landmarks'] = landmarks
            
            # Store landmarks for motion analysis
            self.landmark_history.append(landmarks)
            
            # Calculate EAR for both eyes
            left_ear = self._calculate_ear(landmarks, self.LEFT_EYE)
            right_ear = self._calculate_ear(landmarks, self.RIGHT_EYE)
            ear = (left_ear + right_ear) / 2.0
            
            output['ear_value'] = ear
            self.ear_history.append(ear)
            
            # Blink detection
            if ear < self.blink_threshold:
                output['blink_detected'] = True
                self.blink_counter += 1
            
            # Calculate MAR for lip movement
            mar = self._calculate_mar(landmarks)
            output['mar_value'] = mar
            
            # Head pose estimation
            pose = self._estimate_head_pose(landmarks, frame.shape)
            if pose:
                output['head_pose'] = {
                    'pitch': pose[0],
                    'yaw': pose[1],
                    'roll': pose[2]
                }
                self.pose_history.append(pose)
            
            # Extract face region for texture analysis
            h, w = frame.shape[:2]
            face_bbox = self._get_face_bbox(landmarks, w, h)
            if face_bbox:
                x1, y1, x2, y2 = face_bbox
                face_region = frame[y1:y2, x1:x2]
                output['texture_score'] = self._analyze_texture(face_region)
            
            # Motion analysis
            output['motion_score'] = self._calculate_motion_score()
        
        return output
    
    def _get_face_bbox(self, landmarks, img_w, img_h, padding=0.2):
        """Extract bounding box from landmarks"""
        x_coords = [lm.x * img_w for lm in landmarks]
        y_coords = [lm.y * img_h for lm in landmarks]
        
        x1 = int(min(x_coords))
        x2 = int(max(x_coords))
        y1 = int(min(y_coords))
        y2 = int(max(y_coords))
        
        # Add padding
        w = x2 - x1
        h = y2 - y1
        x1 = max(0, int(x1 - w * padding))
        x2 = min(img_w, int(x2 + w * padding))
        y1 = max(0, int(y1 - h * padding))
        y2 = min(img_h, int(y2 + h * padding))
        
        return (x1, y1, x2, y2)
    
    def analyze_video_sequence(self, frames: List[np.ndarray]) -> Dict:
        """
        Analyze a sequence of video frames for liveness
        
        Args:
            frames: List of BGR images
        
        Returns:
            Comprehensive liveness analysis results
        """
        self.reset()
        
        frame_results = []
        for frame in frames:
            result = self.process_frame(frame)
            frame_results.append(result)
        
        # Aggregate results
        faces_detected = sum(1 for r in frame_results if r['face_detected'])
        blinks_detected = sum(1 for r in frame_results if r['blink_detected'])
        
        avg_ear = np.mean([r['ear_value'] for r in frame_results if r['ear_value'] > 0]) if frame_results else 0
        avg_texture = np.mean([r['texture_score'] for r in frame_results])
        avg_motion = np.mean([r['motion_score'] for r in frame_results if r['motion_score'] > 0]) if frame_results else 0
        
        # Head pose variance (natural head movement)
        pose_variance = 0
        if self.pose_history:
            poses = np.array(list(self.pose_history))
            pose_variance = np.mean(np.var(poses, axis=0))
        
        # Calculate final liveness score
        liveness_score = self._calculate_liveness_score(
            faces_detected / max(len(frames), 1),
            blinks_detected,
            avg_texture,
            avg_motion,
            pose_variance
        )
        
        is_live = liveness_score > 0.6
        
        return {
            'is_live': is_live,
            'liveness_score': liveness_score,
            'total_frames': len(frames),
            'faces_detected': faces_detected,
            'blinks_detected': blinks_detected,
            'average_ear': avg_ear,
            'average_texture_score': avg_texture,
            'average_motion_score': avg_motion,
            'pose_variance': pose_variance,
            'confidence': min(liveness_score, 1.0),
            'reasons': self._get_liveness_reasons(
                faces_detected / max(len(frames), 1),
                blinks_detected,
                avg_texture,
                avg_motion,
                pose_variance
            )
        }
    
    def _calculate_liveness_score(self, face_ratio, blinks, texture, motion, pose_var):
        """Calculate weighted liveness score"""
        scores = []
        weights = []
        
        # Face detection consistency
        scores.append(face_ratio)
        weights.append(0.15)
        
        # Blink detection (at least 1 blink expected in a few seconds)
        blink_score = min(1.0, blinks / 2) if blinks > 0 else 0.3
        scores.append(blink_score)
        weights.append(0.25)
        
        # Texture quality
        scores.append(texture)
        weights.append(0.25)
        
        # Natural motion
        scores.append(motion)
        weights.append(0.20)
        
        # Head pose variation
        pose_score = min(1.0, pose_var * 10) if pose_var > 0 else 0.3
        scores.append(pose_score)
        weights.append(0.15)
        
        return sum(s * w for s, w in zip(scores, weights))
    
    def _get_liveness_reasons(self, face_ratio, blinks, texture, motion, pose_var):
        """Generate human-readable reasons for liveness decision"""
        reasons = []
        
        if face_ratio < 0.5:
            reasons.append("Face not consistently detected")
        
        if blinks == 0:
            reasons.append("No blinks detected - possible photo")
        elif blinks >= 2:
            reasons.append("Natural blink patterns detected")
        
        if texture < 0.4:
            reasons.append("Low texture quality - possible screen/print")
        elif texture > 0.7:
            reasons.append("Good texture quality")
        
        if motion < 0.3:
            reasons.append("Very little movement detected")
        elif motion > 0.5:
            reasons.append("Natural micro-movements detected")
        
        if pose_var > 0.01:
            reasons.append("Natural head movement variation")
        
        return reasons
    
    def reset(self):
        """Reset detector state"""
        self.ear_history.clear()
        self.pose_history.clear()
        self.landmark_history.clear()
        self.blink_counter = 0
        self.frame_count = 0
        self.last_landmarks = None
    
    def get_face_crop(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract cropped face region from frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]
            bbox = self._get_face_bbox(landmarks, w, h, padding=0.3)
            if bbox:
                x1, y1, x2, y2 = bbox
                return frame[y1:y2, x1:x2]
        
        return None


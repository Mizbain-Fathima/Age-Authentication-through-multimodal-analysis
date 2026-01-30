"""
Lip Sync Verification
Verifies that lip movements correlate with spoken audio using MediaPipe Tasks FaceLandmarker
"""
import cv2
import numpy as np
from collections import deque
from typing import List, Dict, Tuple, Optional
from scipy.stats import pearsonr
from loguru import logger
import threading

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.vision import face_landmarker
from mediapipe import Image as MPImage, ImageFormat

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import LIP_SYNC_THRESHOLD


class LipSyncVerifier:
    """
    Verify lip-audio synchronization using MediaPipe Tasks FaceLandmarker
    Analyzes correlation between lip movement and audio energy
    """
    
    # Thread-local storage for MediaPipe instances
    _local = threading.local()
    
    def __init__(self, sync_threshold=LIP_SYNC_THRESHOLD, fps=30, audio_sample_rate=16000):
        self.sync_threshold = sync_threshold
        self.fps = fps
        self.audio_sample_rate = audio_sample_rate
        
        # MediaPipe Tasks model path
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
        
        # Lip landmark indices (MediaPipe 468 landmarks)
        # Upper lip landmarks
        self.UPPER_LIP = [13, 82, 81, 80, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324]
        # Lower lip landmarks
        self.LOWER_LIP = [14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
        
        logger.info("Initialized LipSyncVerifier with MediaPipe Tasks")
    
    @property
    def face_landmarker(self):
        """Get thread-local FaceLandmarker instance"""
        if not hasattr(self._local, 'face_landmarker'):
            self._local.face_landmarker = face_landmarker.FaceLandmarker.create_from_options(
                self.face_landmarker_options
            )
        return self._local.face_landmarker
    
    def _get_model_path(self) -> str:
        """Get path to face landmarker model"""
        project_root = Path(__file__).resolve().parents[2]
        model_path = project_root / "models" / "face_landmarker.task"
        if model_path.exists():
            return str(model_path)
        raise RuntimeError(f"FaceLandmarker model not found at: {model_path}")
    
    def _extract_lip_features(self, frame: np.ndarray) -> Optional[float]:
        """
        Extract mouth opening distance from a frame
        
        Returns:
            Mouth opening distance (float) or None if face not detected
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = MPImage(image_format=ImageFormat.SRGB, data=rgb_frame)
        
        detection_result = self.face_landmarker.detect(mp_image)
        
        if not detection_result.face_landmarks or len(detection_result.face_landmarks) == 0:
            return None
        
        landmarks = detection_result.face_landmarks[0]
        h, w = frame.shape[:2]
        
        # Get upper and lower lip center points
        upper_lip_y = np.mean([landmarks[i].y * h for i in self.UPPER_LIP])
        lower_lip_y = np.mean([landmarks[i].y * h for i in self.LOWER_LIP])
        
        # Calculate mouth opening (vertical distance)
        mouth_opening = abs(upper_lip_y - lower_lip_y)
        
        return mouth_opening
    
    def _extract_audio_energy(self, audio: np.ndarray, num_frames: int) -> np.ndarray:
        """Extract RMS audio energy envelope aligned to video frames"""
        # Calculate samples per video frame
        samples_per_frame = int(self.audio_sample_rate / self.fps)
        
        # Compute RMS energy for each frame
        energy = []
        for i in range(num_frames):
            start = i * samples_per_frame
            end = min((i + 1) * samples_per_frame, len(audio))
            
            if start < len(audio):
                frame_audio = audio[start:end]
                rms = np.sqrt(np.mean(frame_audio ** 2))
                energy.append(rms)
            else:
                energy.append(0.0)
        
        energy = np.array(energy)
        
        # Normalize to [0, 1]
        if np.max(energy) > 0:
            energy = energy / np.max(energy)
        
        return energy
    
    def verify_sync(self, frames: List[np.ndarray], audio: np.ndarray) -> Dict:
        """
        Verify lip-audio synchronization using Pearson correlation
        
        Args:
            frames: List of video frames (BGR)
            audio: Audio waveform at self.audio_sample_rate
        
        Returns:
            Verification results with real correlation score
        """
        if len(frames) < 10:
            return {
                'synced': False,
                'confidence': 0.0,
                'correlation': 0.0,
                'lag_frames': 0,
                'reason': 'Insufficient frames for analysis'
            }
        
        # Extract lip movement (mouth opening) from frames
        lip_movements = []
        valid_frames = 0
        
        for frame in frames:
            mouth_opening = self._extract_lip_features(frame)
            if mouth_opening is not None:
                lip_movements.append(mouth_opening)
                valid_frames += 1
            else:
                # Interpolate if face not detected
                if lip_movements:
                    lip_movements.append(lip_movements[-1])
                else:
                    lip_movements.append(0.0)
        
        lip_movement_arr = np.array(lip_movements)
        
        # Check if enough valid frames
        if valid_frames < len(frames) * 0.5:
            return {
                'synced': False,
                'confidence': 0.0,
                'correlation': 0.0,
                'lag_frames': 0,
                'reason': f'Face detected in only {valid_frames}/{len(frames)} frames'
            }
        
        # Extract audio energy
        audio_energy = self._extract_audio_energy(audio, len(frames))
        
        # Normalize both signals
        if np.std(lip_movement_arr) > 1e-8 and np.std(audio_energy) > 1e-8:
            lip_norm = (lip_movement_arr - np.mean(lip_movement_arr)) / (np.std(lip_movement_arr) + 1e-8)
            audio_norm = (audio_energy - np.mean(audio_energy)) / (np.std(audio_energy) + 1e-8)
            
            # Compute Pearson correlation
            try:
                correlation, p_value = pearsonr(lip_norm, audio_norm)
                correlation = float(correlation) if not np.isnan(correlation) else 0.0
            except Exception as e:
                logger.info(f"Pearson correlation computation failed: {e}")
                correlation = 0.0
        else:
            correlation = 0.0
        
        # Clamp correlation to [0, 1] for score
        # NOTE: Low correlation scores (0.05-0.15) are NORMAL for real humans
        # Reasons: calm speech, minimal mouth movement, neutral expressions,
        # FPS mismatch, clear vowels with steady mouth shape
        # DO NOT inflate or boost these scores - they are honest measurements
        correlation_score = max(0.0, min(1.0, abs(correlation)))
        
        # Determine sync status
        synced = correlation_score > self.sync_threshold
        
        # Calculate confidence (normalized to [0, 1])
        # This is the raw Pearson correlation - low values are expected and correct
        confidence = correlation_score
        
        # Generate reason
        if synced:
            reason = f"Lip movement correlates with audio (r={correlation:.3f})"
        elif correlation_score > self.sync_threshold * 0.5:
            reason = f"Partial correlation detected (r={correlation:.3f})"
        else:
            reason = "Lip movement does not match audio"
        
        # Additional validation: check for lip movement variation
        lip_variance = np.var(lip_movement_arr)
        audio_variance = np.var(audio_energy)
        
        if lip_variance < 0.001 and audio_variance > 0.01:
            synced = False
            confidence = 0.0
            reason = "Lips not moving despite audio"
        
        return {
            'synced': synced,
            'confidence': confidence,
            'correlation': correlation_score,
            'lag_frames': 0,  # Pearson correlation doesn't provide lag
            'lag_ms': 0.0,
            'lip_variance': float(lip_variance),
            'audio_variance': float(audio_variance),
            'valid_face_ratio': valid_frames / len(frames),
            'reason': reason
        }

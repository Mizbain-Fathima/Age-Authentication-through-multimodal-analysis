"""
Lip Sync Verification
Verifies that lip movements correlate with spoken audio
"""
import cv2
import numpy as np
from collections import deque
from typing import List, Dict, Tuple, Optional
from scipy.signal import correlate
from loguru import logger

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import LIP_SYNC_THRESHOLD


class LipSyncVerifier:
    """
    Verify lip-audio synchronization
    Analyzes correlation between lip movement and audio energy
    """
    
    def __init__(self, sync_threshold=LIP_SYNC_THRESHOLD, fps=30, audio_sample_rate=16000):
        self.sync_threshold = sync_threshold
        self.fps = fps
        self.audio_sample_rate = audio_sample_rate
        
        # MediaPipe for lip landmarks
        import mediapipe as mp
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Lip landmark indices (inner lip for better movement detection)
        self.UPPER_LIP = [13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14]
        self.LOWER_LIP = [14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13]
        
        logger.info("Initialized LipSyncVerifier")
    
    def _extract_lip_features(self, frame: np.ndarray) -> Optional[Dict]:
        """Extract lip features from a frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
        
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]
        
        # Get lip coordinates
        upper_lip_pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in self.UPPER_LIP]
        lower_lip_pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in self.LOWER_LIP]
        
        # Calculate lip opening (vertical distance)
        upper_center = np.mean([landmarks[13].y, landmarks[14].y]) * h
        lower_center = np.mean([landmarks[14].y, landmarks[17].y]) * h
        
        lip_distance = abs(landmarks[14].y - landmarks[13].y) * h
        
        # Calculate lip width
        lip_width = abs(landmarks[78].x - landmarks[308].x) * w
        
        # Lip aspect ratio
        lar = lip_distance / (lip_width + 1e-6)
        
        # Lip area (approximate)
        upper_lip_area = cv2.contourArea(np.array(upper_lip_pts, dtype=np.int32))
        lower_lip_area = cv2.contourArea(np.array(lower_lip_pts, dtype=np.int32))
        total_area = upper_lip_area + lower_lip_area
        
        return {
            'lip_distance': lip_distance,
            'lip_width': lip_width,
            'lip_aspect_ratio': lar,
            'lip_area': total_area,
            'upper_lip_points': upper_lip_pts,
            'lower_lip_points': lower_lip_pts
        }
    
    def _extract_audio_energy(self, audio: np.ndarray, num_frames: int) -> np.ndarray:
        """Extract audio energy envelope aligned to video frames"""
        import librosa
        
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
        
        # Normalize
        if np.max(energy) > 0:
            energy = energy / np.max(energy)
        
        return energy
    
    def _compute_correlation(self, lip_movement: np.ndarray, 
                             audio_energy: np.ndarray) -> Tuple[float, int]:
        """
        Compute cross-correlation between lip movement and audio energy
        
        Returns:
            Tuple of (max_correlation, lag_in_frames)
        """
        if len(lip_movement) < 5 or len(audio_energy) < 5:
            return 0.0, 0
        
        # Normalize signals
        lip_norm = (lip_movement - np.mean(lip_movement)) / (np.std(lip_movement) + 1e-8)
        audio_norm = (audio_energy - np.mean(audio_energy)) / (np.std(audio_energy) + 1e-8)
        
        # Zero-pad to same length
        max_len = max(len(lip_norm), len(audio_norm))
        lip_padded = np.pad(lip_norm, (0, max_len - len(lip_norm)), mode='constant')
        audio_padded = np.pad(audio_norm, (0, max_len - len(audio_norm)), mode='constant')
        
        # Cross-correlation
        correlation = correlate(lip_padded, audio_padded, mode='full')
        
        # Find peak
        mid_point = len(correlation) // 2
        # Allow some lag (up to 10 frames = ~333ms at 30fps)
        search_range = 10
        search_region = correlation[mid_point - search_range:mid_point + search_range + 1]
        
        max_corr_idx = np.argmax(np.abs(search_region))
        max_correlation = np.abs(search_region[max_corr_idx])
        lag = max_corr_idx - search_range
        
        # Normalize by signal length
        max_correlation = max_correlation / max_len
        
        return max_correlation, lag
    
    def verify_sync(self, frames: List[np.ndarray], audio: np.ndarray) -> Dict:
        """
        Verify lip-audio synchronization
        
        Args:
            frames: List of video frames (BGR)
            audio: Audio waveform at self.audio_sample_rate
        
        Returns:
            Verification results
        """
        if len(frames) < 10:
            return {
                'synced': False,
                'confidence': 0.0,
                'correlation': 0.0,
                'lag_frames': 0,
                'reason': 'Insufficient frames for analysis'
            }
        
        # Extract lip movement from frames
        lip_movements = []
        valid_frames = 0
        
        for frame in frames:
            features = self._extract_lip_features(frame)
            if features:
                lip_movements.append(features['lip_aspect_ratio'])
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
        
        # Compute correlation
        correlation, lag = self._compute_correlation(lip_movement_arr, audio_energy)
        
        # Determine sync status
        synced = correlation > self.sync_threshold
        
        # Calculate confidence
        confidence = min(1.0, correlation / self.sync_threshold)
        
        # Generate reason
        if synced:
            reason = f"Lip movement correlates with audio (r={correlation:.3f})"
        elif correlation > self.sync_threshold * 0.5:
            reason = f"Partial correlation detected (r={correlation:.3f})"
        else:
            reason = "Lip movement does not match audio"
        
        # Additional analysis: check for lip movement variation
        lip_variance = np.var(lip_movement_arr)
        audio_variance = np.var(audio_energy)
        
        if lip_variance < 0.001 and audio_variance > 0.01:
            synced = False
            reason = "Lips not moving despite audio"
        
        return {
            'synced': synced,
            'confidence': confidence,
            'correlation': correlation,
            'lag_frames': lag,
            'lag_ms': lag * (1000 / self.fps),
            'lip_variance': lip_variance,
            'audio_variance': audio_variance,
            'valid_face_ratio': valid_frames / len(frames),
            'reason': reason
        }
    
    def extract_lip_sequence(self, frames: List[np.ndarray]) -> Dict:
        """
        Extract lip movement sequence for visualization
        
        Args:
            frames: List of video frames
        
        Returns:
            Lip movement data for plotting
        """
        lip_distances = []
        lip_areas = []
        lip_ratios = []
        
        for frame in frames:
            features = self._extract_lip_features(frame)
            if features:
                lip_distances.append(features['lip_distance'])
                lip_areas.append(features['lip_area'])
                lip_ratios.append(features['lip_aspect_ratio'])
            else:
                lip_distances.append(lip_distances[-1] if lip_distances else 0)
                lip_areas.append(lip_areas[-1] if lip_areas else 0)
                lip_ratios.append(lip_ratios[-1] if lip_ratios else 0)
        
        return {
            'lip_distances': np.array(lip_distances),
            'lip_areas': np.array(lip_areas),
            'lip_ratios': np.array(lip_ratios),
            'frame_count': len(frames),
            'time_axis': np.arange(len(frames)) / self.fps
        }


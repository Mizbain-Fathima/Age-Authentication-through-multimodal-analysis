"""
Services for Age Authentication System
Business logic layer between API routes and ML models
"""
import torch
import numpy as np
import cv2
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import uuid
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import DEVICE, MODELS_DIR, FUSION_MODELS_DIR, AGE_GROUPS, AGE_THRESHOLD, ADULT_THRESHOLD, FRAME_SUBSAMPLE_RATE

# Lip landmark indices (MediaPipe 468) for mouth opening from landmarks
_UPPER_LIP = [13, 82, 81, 80, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324]
_LOWER_LIP = [14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
from src.liveness.face_liveness import FaceLivenessDetector
from src.liveness.voice_liveness import VoiceLivenessDetector
from src.liveness.lip_sync import LipSyncVerifier
from src.liveness.eye_blink import EyeBlinkDetector
from src.liveness.captcha import CaptchaSession


class AgeAuthenticationService:
    """
    Main service class for age authentication
    Coordinates between models, liveness detection, and verification
    """
    
    def __init__(self):
        self.device = DEVICE
        
        # Initialize models (will be loaded lazily)
        self.face_model = None
        self.voice_model = None
        self.fusion_model = None
        
        # Initialize liveness detectors
        self.face_liveness = FaceLivenessDetector()
        self.voice_liveness = VoiceLivenessDetector()
        
        # Initialize lip-sync verifier
        try:
            self.lip_sync = LipSyncVerifier()
            logger.info("Lip-sync verifier initialized")
        except Exception as e:
            logger.info(f"Lip-sync unavailable: {e}")
            self.lip_sync = None
        
        # Initialize eye blink detector
        try:
            self.eye_blink = EyeBlinkDetector()
            logger.info("Eye blink detector initialized")
        except Exception as e:
            logger.info(f"Eye blink unavailable: {e}")
            self.eye_blink = None
        
        # Captcha session manager
        self.captcha_session = CaptchaSession()
        
        # Try to load pre-trained models
        self._load_models()
        
        logger.info("AgeAuthenticationService initialized")

    def _mouth_opening_from_landmarks(self, landmarks, h: int) -> Optional[float]:
        """Compute mouth opening from landmark list (items with .x .y)."""
        if not landmarks or len(landmarks) < max(max(_UPPER_LIP), max(_LOWER_LIP)) + 1:
            return None
        try:
            upper_y = np.mean([landmarks[i].y * h for i in _UPPER_LIP if i < len(landmarks)])
            lower_y = np.mean([landmarks[i].y * h for i in _LOWER_LIP if i < len(landmarks)])
            return float(abs(upper_y - lower_y))
        except (IndexError, AttributeError):
            return None

    def _extract_landmarks_for_frames(self, frames: List[np.ndarray]) -> Tuple[List[Dict], List[np.ndarray]]:
        """Run MediaPipe once per frame; return per-frame results and face crops. Each result has face_detected, landmarks (or None), mouth_opening (float or None)."""
        frame_results = []
        face_crops = []
        for frame in frames:
            try:
                result = self.face_liveness.process_frame(frame)
            except Exception:
                result = {"face_detected": False, "landmarks": None, "mouth_opening": None}
            h, w = frame.shape[:2] if frame is not None and len(frame.shape) >= 2 else (0, 0)
            landmarks = result.get("landmarks") if result.get("landmarks") is not None else None
            face_detected = bool(result.get("face_detected", False))
            mouth_opening = self._mouth_opening_from_landmarks(landmarks, h) if landmarks else None
            out = {
                "face_detected": face_detected,
                "landmarks": landmarks,
                "mouth_opening": mouth_opening if mouth_opening is not None else None,
            }
            for k, v in result.items():
                if k not in out:
                    out[k] = v
            frame_results.append(out)
            if face_detected and landmarks is not None:
                try:
                    x_coords = [lm.x * w for lm in landmarks]
                    y_coords = [lm.y * h for lm in landmarks]
                    x1 = int(min(x_coords))
                    x2 = int(max(x_coords))
                    y1 = int(min(y_coords))
                    y2 = int(max(y_coords))
                    pad = 0.2
                    bw, bh = x2 - x1, y2 - y1
                    x1 = max(0, int(x1 - bw * pad))
                    x2 = min(w, int(x2 + bw * pad))
                    y1 = max(0, int(y1 - bh * pad))
                    y2 = min(h, int(y2 + bh * pad))
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        face_crops.append(crop)
                except (IndexError, AttributeError, TypeError):
                    pass
        return frame_results, face_crops

    def _safe_float(self, value) -> float:
        """Convert value to float, replacing NaN/Inf with 0.0"""
        try:
            if value is None:
                return 0.0
            fval = float(value)
            if not np.isfinite(fval):
                return 0.0
            return fval
        except (ValueError, TypeError):
            return 0.0
    
    def _sanitize(self, value):
        """Convert NumPy types to native Python types for JSON serialization"""
        if value is None:
            return None
        if isinstance(value, (np.bool_, bool)):
            return bool(value)
        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            if np.isnan(value) or np.isinf(value):
                return 0.0
            return float(value)
        if isinstance(value, dict):
            return {k: self._sanitize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._sanitize(v) for v in value]
        if isinstance(value, np.ndarray):
            return self._sanitize(value.tolist())
        return value
    
    def _load_models(self):
        """Load pre-trained models if available"""
        # Try to load face model
        face_checkpoint = MODELS_DIR / "face_age_model_best.pth"
        if face_checkpoint.exists():
            try:
                from src.models.face_model import load_face_model
                self.face_model = load_face_model(str(face_checkpoint))
                if self.face_model is not None:
                    self.face_model.eval()
                logger.info("Loaded face model")
            except Exception as e:
                logger.warning(f"Could not load face model: {e}")
        else:
            try:
                from src.models.face_model import create_face_model
                self.face_model = create_face_model(pretrained=True)
                if self.face_model is not None:
                    self.face_model.eval()
                logger.info("Created default face model (not trained)")
            except Exception as e:
                logger.warning(f"Could not create face model: {e}")

        voice_checkpoint = MODELS_DIR / "voice_age_model_best.pth"
        if voice_checkpoint.exists():
            try:
                from src.models.voice_model import load_voice_model
                self.voice_model = load_voice_model(str(voice_checkpoint))
                if self.voice_model is not None:
                    self.voice_model.eval()
                logger.info("Loaded voice model")
            except Exception as e:
                logger.warning(f"Could not load voice model: {e}")
        else:
            try:
                from src.models.voice_model import create_voice_model
                self.voice_model = create_voice_model()
                if self.voice_model is not None:
                    self.voice_model.eval()
                logger.info("Created default voice model (not trained)")
            except Exception as e:
                logger.warning(f"Could not create voice model: {e}")

        from src.config import USE_FUSION_FOR_AGE
        if USE_FUSION_FOR_AGE:
            # Try models/fusion_age_model_best.pth first, then models/fusion/fusion_age_model_best.pth
            fusion_checkpoint = MODELS_DIR / "fusion_age_model_best.pth"
            if not fusion_checkpoint.exists():
                fusion_checkpoint = FUSION_MODELS_DIR / "fusion_age_model_best.pth"
            if not fusion_checkpoint.exists():
                logger.warning(
                    f"USE_FUSION_FOR_AGE is True but fusion checkpoint not found. "
                    "Tried: models/fusion_age_model_best.pth and models/fusion/fusion_age_model_best.pth. "
                    "Train with: python run.py train fusion."
                )
            else:
                try:
                    from src.models.multimodal_model import load_multimodal_model
                    self.fusion_model = load_multimodal_model(str(fusion_checkpoint))
                    if self.fusion_model is not None:
                        self.fusion_model.eval()
                        logger.info(f"Loaded fusion model from {fusion_checkpoint}")
                    else:
                        logger.warning("load_multimodal_model returned None")
                except Exception as e:
                    logger.warning(f"Could not load fusion model: {e}", exc_info=True)
        else:
            logger.info("Age estimation: using face + voice combined (fusion model disabled for speed/stability)")
    
    def generate_captcha(self, complexity: str = "medium") -> Dict:
        """Generate a new captcha for verification"""
        return self.captcha_session.create_session(complexity)
    
    def verify(self, captcha_id: str, frames: Optional[List[np.ndarray]] = None,
               audio: Optional[np.ndarray] = None, sample_rate: int = 16000) -> Dict:
        """
        Complete verification with liveness and age estimation
        
        Args:
            captcha_id: Captcha session ID
            frames: List of video frames (BGR)
            audio: Audio waveform
            sample_rate: Audio sample rate
        
        Returns:
            Complete verification results
        """
        verification_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Validate captcha session
        is_valid, expected_text = self.captcha_session.validate_session(captcha_id)
        if not is_valid:
            response = {
                'success': False,
                'verification_id': verification_id,
                'timestamp': timestamp,
                'liveness': {
                    'is_live': False,
                    'face_liveness_score': 0.0,
                    'voice_liveness_score': 0.0,
                    'lip_sync_score': 0.0,
                    'captcha_verified': False,
                    'reasons': [expected_text]  # Error message
                },
                'age': {
                    'estimated_age': 0,
                    'face_age': None,
                    'voice_age': None,
                    'age_group': 'unknown',
                    'age_group_confidence': 0.0,
                    'is_adult': False,
                    'adult_confidence': 0.0
                },
                'overall_confidence': 0.0,
                'message': expected_text
            }
            return self._sanitize(response)
        
        # Face liveness detection
        face_liveness_result = {'is_live': False, 'liveness_score': 0, 'reasons': []}
        if frames and len(frames) > 0:
            face_liveness_result = self.face_liveness.analyze_video_sequence(frames)
        
        # Voice liveness (captcha verification)
        voice_liveness_result = {'verified': False, 'confidence': 0, 'reason': 'No audio', 'transcription': ''}
        if audio is not None:
            voice_liveness_result = self.voice_liveness.verify_captcha(
                audio, expected_text, sample_rate
            )
        
        # Eye blink detection
        eye_blink_score = None
        faces_detected = face_liveness_result.get('faces_detected', 0)
        if faces_detected > 0 and frames and len(frames) > 0 and self.eye_blink is not None:
            try:
                blink_result = self.eye_blink.detect_blinks(frames)
                eye_blink_score = blink_result.get('blink_score')
                if eye_blink_score is not None:
                    eye_blink_score = self._safe_float(eye_blink_score)
                    eye_blink_score = max(0.0, min(1.0, eye_blink_score))
                    logger.info(f"Eye blink liveness score: {eye_blink_score:.2f} ({blink_result.get('blink_count', 0)} blinks)")
                else:
                    logger.info("Eye blink unavailable: no face detected in frames")
            except Exception as e:
                logger.info(f"Eye blink unavailable: {e}")
                eye_blink_score = None
        else:
            if faces_detected == 0:
                logger.info("Eye blink skipped: no face detected")
            elif not frames or len(frames) == 0:
                logger.info("Eye blink skipped: no video frames available")
            else:
                logger.info("Eye blink unavailable (detector not initialized)")
            eye_blink_score = None
        
        # Lip sync verification
        lip_sync_score = None
        lip_sync_result = {'synced': False, 'confidence': 0, 'reason': 'Lip sync unavailable'}
        if frames and len(frames) > 0 and audio is not None and self.lip_sync is not None:
            try:
                lip_sync_result = self.lip_sync.verify_sync(frames, audio)
                lip_sync_score = self._safe_float(lip_sync_result.get('confidence', 0))
                lip_sync_score = max(0.0, min(1.0, lip_sync_score)) if lip_sync_score is not None else None
            except Exception as e:
                logger.info(f"Lip-sync unavailable: {e}")
                lip_sync_score = None
        
        # Age prediction
        age_result = self._predict_age(frames, audio, sample_rate)
        
        # No demo fallback - captcha must match strictly
        
        # Combine liveness scores
        face_live_score = face_liveness_result.get('liveness_score', 0)
        voice_live_score = voice_liveness_result.get('confidence', 0)
        
        # Overall liveness
        is_live = (
            face_liveness_result.get('is_live', False) and
            voice_liveness_result.get('verified', False)
        )
        
        # Collect reasons
        reasons = []
        if face_liveness_result.get('reasons'):
            reasons.extend(face_liveness_result['reasons'])
        reasons.append(voice_liveness_result.get('reason', ''))
        reasons.append(lip_sync_result.get('reason', ''))
        reasons = [r for r in reasons if r]  # Remove empty strings
        
        # Calculate overall confidence with updated weights
        # face_liveness → 0.45, eye_blink → 0.15 (if available), lip_sync → 0.25 (if available), voice_liveness → 0.15 (if available)
        # NOTE: Low eye_blink (0.4) and lip_sync (0.05-0.15) scores are BY DESIGN and CORRECT
        # - Eye blink: 1 blink = 0.4 (40%) is expected for short recordings
        # - Lip-sync: 0.05-0.15 correlation is normal for calm/slow speech
        # These are CONFIDENCE signals only - they NEVER gate success
        # DO NOT inflate, boost, or add fallbacks - low scores are honest measurements
        confidence_components = []
        face_age_val = age_result.get('face_age')
        voice_age_val = age_result.get('voice_age')
        if face_age_val is not None:
            confidence_components.append(face_live_score * 0.45)
        if eye_blink_score is not None and eye_blink_score > 0:
            confidence_components.append(eye_blink_score * 0.15)
        if lip_sync_score is not None and lip_sync_score > 0:
            confidence_components.append(lip_sync_score * 0.25)
        # Voice liveness (0.15 weight ONLY if voice_age available)
        if voice_age_val is not None:
            confidence_components.append(voice_live_score * 0.15)
        
        overall_confidence = self._safe_float(sum(confidence_components) if confidence_components else 0.0)
        # Clamp confidence to [0.0, 1.0] - NO NaN / Inf allowed
        overall_confidence = max(0.0, min(1.0, overall_confidence))
        
        # Success criteria - strict rule
        # Success if: captcha verified AND face_age IS NOT None
        captcha_verified = voice_liveness_result.get('verified', False)
        success = (
            captcha_verified and
            face_age_val is not None
        )
        
        # Complete captcha session
        if success:
            self.captcha_session.complete_session(captcha_id)
        
        response = {
            'success': success,
            'verification_id': verification_id,
            'timestamp': timestamp,
            'liveness': {
                'is_live': is_live,
                'face_liveness_score': face_live_score,
                'voice_liveness_score': voice_live_score,
                'lip_sync_score': self._safe_float(lip_sync_score) if lip_sync_score is not None else None,
                'eye_blink_score': self._safe_float(eye_blink_score) if eye_blink_score is not None else None,
                'captcha_verified': voice_liveness_result.get('verified', False),
                'reasons': reasons
            },
            'age': {
                'estimated_age': age_result.get('age', 0),
                'face_age': age_result.get('face_age'),
                'voice_age': age_result.get('voice_age'),
                'age_group': age_result.get('age_group', 'unknown'),
                'age_group_confidence': age_result.get('age_group_confidence', 0),
                'is_adult': age_result.get('is_adult', False),
                'adult_confidence': age_result.get('adult_confidence', 0)
            },
            'overall_confidence': overall_confidence,
            'message': 'Verification successful' if success else 'Verification failed'
        }
        
        # Sanitize all NumPy types before returning
        return self._sanitize(response)
    
    def _predict_age(self, frames: Optional[List[np.ndarray]], 
                     audio: Optional[np.ndarray], sample_rate: int) -> Dict:
        """
        Predict APPARENT age from face and/or voice with temporal smoothing
        
        NOTE: The model predicts APPARENT age (how old someone looks), not chronological age.
        Underestimation for ages above ~35 is expected due to:
        - Dataset bias
        - Regression mean collapse
        - Apparent-age learning
        
        This is NOT a bug - it's a feature of apparent-age models.
        Underestimation is acceptable and safe for age verification purposes.
        """
        face_age = None
        voice_age = None
        age_probs = None
        
        # Face age prediction with temporal smoothing (median over N frames)
        if frames and len(frames) > 0 and self.face_model is not None:
            try:
                # Collect age predictions over N frames (N = 15-25)
                N = min(20, len(frames))
                frame_indices = np.linspace(0, len(frames) - 1, N, dtype=int)
                age_predictions = []
                
                for idx in frame_indices:
                    try:
                        face_result = self.predict_face_age(frames[idx])
                        if 'error' not in face_result:
                            pred_age = face_result.get('age')
                            if pred_age is not None and pred_age > 0:
                                age_predictions.append(pred_age)
                                if age_probs is None:
                                    age_probs = face_result.get('age_group_probs')
                    except Exception:
                        continue  # Skip failed predictions
                
                if age_predictions:
                    # Use median (NOT mean) for stability
                    median_age = float(np.median(age_predictions))
                    # Clamp to ±3 years max deviation from median
                    age_std = np.std(age_predictions)
                    if age_std > 3.0:
                        # Filter outliers beyond ±3 years
                        filtered_ages = [a for a in age_predictions if abs(a - median_age) <= 3.0]
                        if filtered_ages:
                            face_age = float(np.median(filtered_ages))
                        else:
                            face_age = median_age
                    else:
                        face_age = median_age
                    
                    # NOTE: face_age is APPARENT age (how old someone looks), not chronological age
                    # Underestimation for ages above ~35 is expected and acceptable
                    # This is NOT a bug - it's a feature of apparent-age models
                else:
                    face_age = None
                    age_probs = None
            except Exception as e:
                logger.info(f"Face age prediction failed: {e}")
                face_age = None
                age_probs = None
        
        # Voice age prediction - skip if audio < 3 seconds
        if audio is not None and self.voice_model is not None:
            audio_duration = len(audio) / sample_rate if sample_rate > 0 else 0
            if audio_duration < 3.0:
                logger.info(f"Voice age skipped: audio too short ({audio_duration:.1f}s < 3s)")
                voice_age = None
            else:
                try:
                    voice_result = self.predict_voice_age(audio, sample_rate)
                    if 'error' not in voice_result:
                        voice_age = voice_result.get('age')
                        if age_probs is None:
                            age_probs = voice_result.get('age_group_probs')
                    else:
                        voice_age = None
                except Exception as e:
                    logger.info(f"Voice age prediction failed (non-critical): {e}")
                    voice_age = None
        
        # Fuse predictions - use face_age if available, voice_age only for confidence
        # NOTE: final_age represents APPARENT age (how old someone looks), not chronological age
        # Underestimation for ages above ~35 is expected and acceptable for age verification purposes
        if face_age is not None:
            final_age = face_age  # Use face age as primary
        elif voice_age is not None:
            final_age = voice_age  # Fallback to voice if face unavailable
        else:
            final_age = 0
        
        # Sanitize final_age
        final_age = self._safe_float(final_age)
        
        # Determine age group
        age_group = self._get_age_group(final_age)
        
        # Is adult (configurable threshold)
        th = float(ADULT_THRESHOLD)
        is_adult = final_age >= th
        adult_confidence = self._safe_float(min(1.0, abs(final_age - th) / 10 + 0.5))
        
        return {
            'age': final_age,
            'face_age': face_age,
            'voice_age': voice_age,
            'age_group': age_group,
            'age_group_confidence': 0.8 if age_probs is not None else 0.5,
            'is_adult': is_adult,
            'adult_confidence': adult_confidence,
            'confidence': 0.7 if (face_age or voice_age) else 0.0
        }
    
    def predict_face_age(self, image: np.ndarray) -> Dict:
        """Predict age from a single face image"""
        if self.face_model is None:
            return {'age': 0, 'error': 'Face model not loaded'}
        
        # Detect and crop face
        face_crop = self.face_liveness.get_face_crop(image)
        if face_crop is None:
            return {'age': 0, 'error': 'No face detected'}
        
        # Preprocess
        from torchvision import transforms
        from src.config import FACE_IMAGE_SIZE
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(FACE_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        input_tensor = transform(face_rgb).unsqueeze(0).to(self.device)
        
        # Predict
        self.face_model.eval()
        with torch.no_grad():
            output = self.face_model(input_tensor)
        
        age = output['age'].item()
        age_group_probs = torch.softmax(output['age_group_logits'], dim=1)[0].cpu().numpy()
        is_adult_prob = output['is_adult'].item()
        
        age_groups = ['child', 'teen', 'adult', 'senior']
        predicted_group = age_groups[np.argmax(age_group_probs)]
        
        return {
            'age': self._safe_float(age),
            'age_group': predicted_group,
            'age_group_probs': {g: self._safe_float(p) for g, p in zip(age_groups, age_group_probs)},
            'is_adult': is_adult_prob > 0.5,
            'is_adult_prob': self._safe_float(is_adult_prob)
        }
    
    def predict_voice_age(self, audio: np.ndarray, sample_rate: int = 16000) -> Dict:
        """Predict age from audio"""
        if self.voice_model is None:
            return {'age': 0, 'error': 'Voice model not loaded'}
        
        # Extract features
        from src.data.audio_dataset import LiveAudioProcessor
        
        processor = LiveAudioProcessor()
        features = processor.process_audio_bytes(audio, sample_rate).to(self.device)
        
        # Predict
        self.voice_model.eval()
        with torch.no_grad():
            output = self.voice_model(features)
        
        age = output['age'].item()
        age_group_probs = torch.softmax(output['age_group_logits'], dim=1)[0].cpu().numpy()
        is_adult_prob = output['is_adult'].item()
        
        age_groups = ['child', 'teen', 'adult', 'senior']
        predicted_group = age_groups[np.argmax(age_group_probs)]
        
        return {
            'age': self._safe_float(age),
            'age_group': predicted_group,
            'age_group_probs': {g: self._safe_float(p) for g, p in zip(age_groups, age_group_probs)},
            'is_adult': is_adult_prob > 0.5,
            'is_adult_prob': self._safe_float(is_adult_prob)
        }

    def _get_audio_embedding_for_multimodal(self, audio: np.ndarray, sample_rate: int):
        """Build flattened audio embedding (same format as MultimodalDataset) for the multimodal model."""
        from src.data.audio_dataset import AudioFeatureExtractor
        from src.config import SAMPLE_RATE, N_MELS
        import librosa
        extractor = AudioFeatureExtractor()
        if sample_rate != extractor.sample_rate:
            y = librosa.resample(audio.astype(np.float64), orig_sr=sample_rate, target_sr=extractor.sample_rate)
        else:
            y = audio.astype(np.float64)
        y, _ = librosa.effects.trim(y, top_db=20)
        max_samples = int(extractor.max_length * extractor.sample_rate)
        if len(y) > max_samples:
            y = y[:max_samples]
        elif len(y) < max_samples:
            y = np.pad(y, (0, max_samples - len(y)), mode='constant')
        mfcc = extractor.extract_mfcc(y)
        mel = extractor.extract_mel_spectrogram(y)
        mfcc_pad = np.zeros((N_MELS, mfcc.shape[1]), dtype=mfcc.dtype)
        mfcc_pad[: mfcc.shape[0], :] = mfcc
        combined = np.stack([mfcc_pad, mel], axis=0)
        return torch.tensor(combined.flatten(), dtype=torch.float32).unsqueeze(0).to(self.device)

    def _predict_multimodal_age(self, frame: np.ndarray, audio: np.ndarray, sample_rate: int) -> Optional[Dict]:
        """Run the trained multimodal model; returns {'age': float, 'is_adult': bool} or None on failure."""
        if self.fusion_model is None or len(audio) == 0:
            return None
        face_crop = self.face_liveness.get_face_crop(frame)
        if face_crop is None:
            return None
        from torchvision import transforms
        from src.config import FACE_IMAGE_SIZE
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(FACE_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        image_tensor = transform(face_rgb).unsqueeze(0).to(self.device)
        audio_tensor = self._get_audio_embedding_for_multimodal(audio, sample_rate)
        self.fusion_model.eval()
        with torch.no_grad():
            out = self.fusion_model(image_tensor, audio_tensor)
        age = out['age'].item()
        is_adult = torch.sigmoid(out['is_adult_logit']).item() > 0.5
        return {'age': self._safe_float(age), 'is_adult': is_adult}
    
    def check_face_liveness(self, frames: List[np.ndarray]) -> Dict:
        """Check face liveness from video frames"""
        result = self.face_liveness.analyze_video_sequence(frames)
        return result
    
    def check_voice_liveness(self, captcha_id: str, audio: np.ndarray,
                             sample_rate: int = 16000) -> Dict:
        """Check voice liveness via captcha verification"""
        # Validate captcha
        is_valid, expected_text = self.captcha_session.validate_session(captcha_id)
        if not is_valid:
            return {
                'verified': False,
                'error': expected_text
            }
        
        # Verify
        result = self.voice_liveness.verify_captcha(audio, expected_text, sample_rate)
        
        # Add voice characteristics
        voice_chars = self.voice_liveness.analyze_voice_characteristics(audio, sample_rate)
        result['voice_characteristics'] = voice_chars
        
        return result
    
    def _get_age_group(self, age: float) -> str:
        """Map numeric age to age group"""
        for group, (min_age, max_age) in AGE_GROUPS.items():
            if min_age <= age <= max_age:
                return group
        return 'adult'
    
    def _calculate_overall_confidence(self, face_live: float, voice_live: float,
                                       lip_sync: float, age_conf: float) -> float:
        """Calculate overall verification confidence"""
        # Weighted combination
        weights = {
            'face_live': 0.3,
            'voice_live': 0.35,
            'lip_sync': 0.15,
            'age': 0.2
        }
        
        confidence = (
            weights['face_live'] * face_live +
            weights['voice_live'] * voice_live +
            weights['lip_sync'] * lip_sync +
            weights['age'] * age_conf
        )
        
        return min(1.0, max(0.0, confidence))
    
    def get_model_status(self) -> Dict:
        """Get status of loaded models"""
        return {
            'face_model': {
                'loaded': self.face_model is not None,
                'device': str(self.device)
            },
            'voice_model': {
                'loaded': self.voice_model is not None,
                'device': str(self.device)
            },
            'fusion_model': {
                'loaded': self.fusion_model is not None,
                'device': str(self.device)
            },
            'liveness_detectors': {
                'face': True,
                'voice': True,
                'lip_sync': True
            }
        }
    
    def process_authentication(
        self,
        frames: List[np.ndarray],
        audio: np.ndarray,
        sample_rate: int = 16000,
        duration: Optional[float] = None,
        expected_captcha_text: Optional[str] = None,
        captcha_id: Optional[str] = None
    ) -> Dict:
        """
        Simplified authentication process
        
        Args:
            frames: List of video frames
            audio: Audio waveform
            sample_rate: Audio sample rate
            duration: Recording duration (optional)
            expected_captcha_text: Expected captcha text for verification
            captcha_id: Captcha session ID
        
        Returns:
            Simplified verification results
        """
        verification_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Use provided captcha text or generate new one (fallback)
        if expected_captcha_text:
            expected_text = expected_captcha_text
        else:
            captcha_data = self.captcha_session.create_session('medium')
            expected_text = captcha_data['sentence']

        # Frame subsampling: only for longer videos to avoid empty or over-aggressive subsample
        rate = max(1, int(FRAME_SUBSAMPLE_RATE))
        if rate > 1 and len(frames) > 10:
            subsampled = list(frames[::rate])
            if len(subsampled) > 0:
                frames = subsampled
        video_too_short = len(frames) < 5

        # Single-pass landmark extraction for video pipeline
        precomputed_list, face_crops = self._extract_landmarks_for_frames(frames)
        mouth_openings = [r.get("mouth_opening") for r in precomputed_list]

        def video_task():
            face_liveness_score = 0.0
            face_age = None
            faces_detected = 0
            blink_detected = False
            head_movement_detected = False
            try:
                face_liveness_result, blink_info, head_movement, _ = self.face_liveness.analyze_comprehensive(
                    frames, precomputed=precomputed_list, face_crops=face_crops
                )
                faces_detected = face_liveness_result.get('faces_detected', 0)
                if faces_detected > 0 and face_crops:
                    face_liveness_score = self._safe_float(face_liveness_result.get('liveness_score', 0))
                    blink_detected = blink_info.get('blink_count', 0) > 0
                    head_movement_detected = head_movement.get('motion_detected', False)
                    if self.face_model is not None and frames:
                        try:
                            face_result = self.predict_face_age(frames[-1])
                            if 'error' not in face_result:
                                face_age = self._safe_float(face_result.get('age'))
                        except Exception as e:
                            logger.info(f"Face age prediction failed: {e}")
                else:
                    logger.info("No faces detected in frames")
            except Exception as e:
                logger.info(f"Face analysis failed: {e}")
            return face_liveness_score, face_age, faces_detected, blink_detected, head_movement_detected

        def audio_task():
            voice_liveness_score = 0.0
            voice_age = None
            captcha_verified = False
            transcription = ""
            if len(audio) > 0 and expected_text:
                try:
                    voice_liveness_result = self.voice_liveness.verify_captcha(audio, expected_text, sample_rate)
                    voice_liveness_score = self._safe_float(voice_liveness_result.get('confidence', 0))
                    captcha_verified = voice_liveness_result.get('verified', False)
                    transcription = voice_liveness_result.get('transcription', '')
                    # Run voice age whenever we have audio and voice model so we get combined 0.6*face+0.4*voice (better than face-only)
                    if self.voice_model is not None:
                        try:
                            voice_result = self.predict_voice_age(audio, sample_rate)
                            if 'error' not in voice_result:
                                voice_age = self._safe_float(voice_result.get('age'))
                                if voice_age is not None:
                                    logger.info(f"Voice age: {voice_age:.1f}")
                            else:
                                logger.info("Voice age skipped: prediction returned error")
                        except Exception as e:
                            logger.info(f"Voice age prediction failed (non-critical): {e}")
                    else:
                        logger.info("Voice age unavailable: voice model not loaded (check models/voice_age_model_best.pth)")
                except Exception as e:
                    logger.info(f"Voice processing failed: {e}")
            return voice_liveness_score, voice_age, captcha_verified, transcription

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_v = executor.submit(video_task)
            future_a = executor.submit(audio_task)
            try:
                face_liveness_score, face_age, faces_detected, blink_detected, head_movement_detected = future_v.result()
            except Exception as e:
                logger.error(f"Video task failed, using face-only fallback: {e}")
                face_liveness_score, face_age, faces_detected, blink_detected, head_movement_detected = 0.0, None, 0, False, False
            try:
                voice_liveness_score, voice_age, captcha_verified, transcription = future_a.result()
            except Exception as e:
                logger.error(f"Audio task failed, using safe defaults: {e}")
                voice_liveness_score, voice_age, captcha_verified, transcription = 0.0, None, False, ""

        # Very short video: skip blink and lip-sync (meaningless with <5 frames)
        if video_too_short:
            logger.warning("Video too short for blink/lip-sync analysis")
            blink_detected = False
        eye_blink_score = None
        if video_too_short:
            eye_blink_score = None
        elif faces_detected > 0 and len(frames) > 0 and self.eye_blink is not None:
            try:
                blink_result = self.eye_blink.detect_blinks(frames=frames, precomputed=precomputed_list)
                eye_blink_score = blink_result.get('blink_score')
                if eye_blink_score is not None:
                    eye_blink_score = self._safe_float(eye_blink_score)
                    eye_blink_score = max(0.0, min(1.0, eye_blink_score))
                    logger.info(f"Eye blink liveness score: {eye_blink_score:.2f} ({blink_result.get('blink_count', 0)} blinks)")
                else:
                    logger.info("Eye blink unavailable: no face detected in frames")
            except Exception as e:
                logger.info(f"Eye blink unavailable: {e}")
                eye_blink_score = None
        else:
            if faces_detected == 0:
                logger.info("Eye blink skipped: no face detected")
            elif len(frames) == 0:
                logger.info("Eye blink skipped: no video frames available")
            else:
                logger.info("Eye blink unavailable (detector not initialized)")
            eye_blink_score = None

        lip_sync_score = None
        if video_too_short:
            lip_sync_score = 0.0
        else:
            audio_duration = len(audio) / sample_rate if sample_rate > 0 else 0
            if faces_detected > 0 and audio_duration >= 3.0 and self.lip_sync is not None:
                try:
                    lip_sync_result = self.lip_sync.verify_sync(
                        frames, audio, precomputed_mouth_openings=mouth_openings
                    )
                    lip_sync_score = self._safe_float(lip_sync_result.get('confidence', 0))
                    lip_sync_score = max(0.0, min(1.0, lip_sync_score))
                    logger.info(f"Lip-sync liveness score: {lip_sync_score:.2f}")
                except Exception as e:
                    logger.info(f"Lip-sync unavailable: {e}")
                    lip_sync_score = None
            else:
                if audio_duration < 3.0:
                    logger.info(f"Lip-sync skipped: audio too short ({audio_duration:.1f}s < 3s)")
                elif faces_detected == 0:
                    logger.info("Lip-sync skipped: no face detected")
                else:
                    logger.info("Lip-sync unavailable (verifier not initialized)")
                lip_sync_score = None
        
        # 6. Age estimation: use fusion model only if enabled; otherwise use face + voice combined (faster, ±2–4 yr behavior)
        estimated_age = None
        is_adult = False
        age_source = None  # "fusion" | "combined" | "face" | "voice"
        from src.config import USE_FUSION_FOR_AGE
        if (
            USE_FUSION_FOR_AGE
            and self.fusion_model is not None
            and faces_detected > 0
            and len(frames) > 0
            and len(audio) > 0
            and captcha_verified
        ):
            try:
                multimodal_result = self._predict_multimodal_age(frames[-1], audio, sample_rate)
                if multimodal_result is not None:
                    estimated_age = multimodal_result['age']
                    is_adult = multimodal_result['is_adult']
                    age_source = "fusion"
                    logger.info(f"Multimodal model: age={estimated_age:.1f}, is_adult={is_adult}")
            except Exception as e:
                logger.warning(f"Multimodal prediction failed, using fallback: {e}")
        if estimated_age is None:
            if face_age is not None and voice_age is not None:
                estimated_age = self._safe_float(0.6 * face_age + 0.4 * voice_age)
                age_source = "combined"
            elif face_age is not None:
                estimated_age = face_age
                age_source = "face"
            elif voice_age is not None:
                estimated_age = voice_age
                age_source = "voice"
            else:
                estimated_age = None
            if estimated_age is not None:
                is_adult = estimated_age >= float(ADULT_THRESHOLD)

        # 7. Calculate confidence with updated weights
        # face_liveness → 0.45, eye_blink → 0.15 (if available), lip_sync → 0.25 (if available), voice_liveness → 0.15 (if available)
        # NOTE: Low eye_blink (0.4) and lip_sync (0.05-0.15) scores are BY DESIGN and CORRECT
        # - Eye blink: 1 blink = 0.4 (40%) is expected for short recordings
        # - Lip-sync: 0.05-0.15 correlation is normal for calm/slow speech
        # These are CONFIDENCE signals only - they NEVER gate success
        # DO NOT inflate, boost, or add fallbacks - low scores are honest measurements
        confidence_components = []
        
        # Face liveness (primary, 0.45 weight)
        if face_age is not None:
            confidence_components.append(face_liveness_score * 0.45)
        
        # Eye blink (0.15 weight if available)
        # Accepts low scores (0.4 for 1 blink) - this is CORRECT behavior
        if eye_blink_score is not None and eye_blink_score > 0:
            confidence_components.append(eye_blink_score * 0.15)
        
        # Lip-sync (0.25 weight if available)
        # Accepts low scores (0.05-0.15) - this is NORMAL for real humans
        if lip_sync_score is not None and lip_sync_score > 0:
            confidence_components.append(lip_sync_score * 0.25)
        
        # Voice liveness (0.15 weight ONLY if voice_age available)
        if voice_age is not None:
            confidence_components.append(voice_liveness_score * 0.15)
        
        confidence = self._safe_float(sum(confidence_components) if confidence_components else 0.0)
        # Clamp confidence to [0.0, 1.0] - NO NaN / Inf allowed
        confidence = max(0.0, min(1.0, confidence))
        
        # 8. Success determination - strict rule
        # Success if: captcha verified AND face_age IS NOT None
        success = (
            captcha_verified and
            face_age is not None
        )
        
        # 7. Build response with sanitized values
        response = {
            'success': success,
            'estimated_age': self._safe_float(estimated_age) if estimated_age is not None else None,
            'is_adult': is_adult,
            'age_source': age_source,
            'face_age': self._safe_float(face_age) if face_age is not None else None,
            'voice_age': self._safe_float(voice_age) if voice_age is not None else None,
            'confidence': max(0.0, min(1.0, confidence)),  # Clamp to [0, 1]
            'checks': {
                'face_detected': faces_detected > 0,
                'face_liveness': max(0.0, min(1.0, face_liveness_score)),  # Clamp to [0, 1]
                'blink_detected': blink_detected,
                'head_movement': head_movement_detected,
                'captcha_verified': captcha_verified,
                'voice_liveness': max(0.0, min(1.0, voice_liveness_score)),  # Clamp to [0, 1]
                'lip_sync': self._safe_float(lip_sync_score) if lip_sync_score is not None else None, # Return null if unavailable
                'eye_blink': self._safe_float(eye_blink_score) if eye_blink_score is not None else None # Return null if unavailable
            },
            'message': 'Verification successful' if success else 'Verification failed'
        }
        
        # Memory cleanup for large frame lists
        try:
            del frames
        except NameError:
            pass
        try:
            del precomputed_list
        except NameError:
            pass
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        return self._sanitize(response)

    def process_live_authentication(
        self,
        captcha_id: str,
        session_id: str,
        frames: List[np.ndarray],
        audio: np.ndarray,
        sample_rate: int = 16000,
        duration: Optional[float] = None
    ) -> Dict:
        """
        Complete live authentication with comprehensive analysis
        
        Args:
            captcha_id: Captcha session ID
            session_id: Process session ID
            frames: List of video frames
            audio: Audio waveform
            sample_rate: Audio sample rate
            duration: Recording duration (optional)
        
        Returns:
            Complete authentication results with detailed analysis
        """
        verification_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # 1. Validate captcha session
        is_valid, expected_text = self.captcha_session.validate_session(captcha_id)
        if not is_valid:
            return self._create_failed_response(
                verification_id, timestamp, session_id, expected_text
            )
        
        # 2. Comprehensive face analysis in single pass (optimized)
        try:
            face_liveness_result, blink_info, head_movement, face_crops = \
                self.face_liveness.analyze_comprehensive(frames)
            
            # Guard: Check if no faces detected
            if face_liveness_result.get('faces_detected', 0) == 0 or len(face_crops) == 0:
                logger.info("No faces detected in frames")
                face_liveness_result = {
                    'is_live': False,
                    'liveness_score': 0.0,
                    'total_frames': len(frames),
                    'faces_detected': 0,
                    'blinks_detected': 0,
                    'reasons': ['No faces detected in frames']
                }
                blink_info = {
                    'blink_count': 0,
                    'blink_rate': 0.0,
                    'is_valid': False,
                    'blink_score': 0.0,
                    'blink_intervals': []
                }
                head_movement = {
                    'motion_detected': False,
                    'motion_score': 0.0,
                    'movement_pattern': 'no_faces',
                    'head_pose_changes': []
                }
                face_crops = []
        except Exception as e:
            logger.info(f"Comprehensive face analysis failed: {e}")
            face_liveness_result = {
                'is_live': False,
                'liveness_score': 0.0,
                'total_frames': len(frames),
                'faces_detected': 0,
                'blinks_detected': 0,
                'reasons': [f"Face analysis error: {str(e)}"]
            }
            blink_info = {
                'blink_count': 0,
                'blink_rate': 0.0,
                'is_valid': False,
                'blink_score': 0.0,
                'blink_intervals': []
            }
            head_movement = {
                'motion_detected': False,
                'motion_score': 0.0,
                'movement_pattern': 'error',
                'head_pose_changes': []
            }
            face_crops = []
        
        # 3. Voice liveness (captcha verification) (with error handling)
        try:
            voice_liveness_result = self.voice_liveness.verify_captcha(
                audio, expected_text, sample_rate
            )
        except Exception as e:
            logger.info(f"Voice liveness verification failed: {e}")
            voice_liveness_result = {
                'verified': False,
                'confidence': 0.0,
                'match_score': 0.0,
                'reason': f"Voice verification error: {str(e)}",
                'speech_detected': False,
                'transcription': ''
            }
        
        # 4. Eye blink detection (with error handling)
        # Run ONLY when: face detected AND video frames available
        eye_blink_score = None
        faces_detected = len(face_crops) if face_crops else 0
        if faces_detected > 0 and len(frames) > 0 and self.eye_blink is not None:
            try:
                blink_result = self.eye_blink.detect_blinks(frames)
                eye_blink_score = blink_result.get('blink_score')
                if eye_blink_score is not None:
                    eye_blink_score = self._safe_float(eye_blink_score)
                    eye_blink_score = max(0.0, min(1.0, eye_blink_score))
                    logger.info(f"Eye blink liveness score: {eye_blink_score:.2f} ({blink_result.get('blink_count', 0)} blinks)")
                else:
                    logger.info("Eye blink unavailable: no face detected in frames")
            except Exception as e:
                logger.info(f"Eye blink unavailable: {e}")
                eye_blink_score = None
        else:
            if faces_detected == 0:
                logger.info("Eye blink skipped: no face detected")
            elif len(frames) == 0:
                logger.info("Eye blink skipped: no video frames available")
            else:
                logger.info("Eye blink unavailable (detector not initialized)")
            eye_blink_score = None
        
        # 5. Lip-sync verification (with error handling)
        # Run ONLY when: face detected AND audio duration >= 3 seconds
        lip_sync_score = None
        audio_duration = len(audio) / sample_rate if sample_rate > 0 else 0
        
        if faces_detected > 0 and audio_duration >= 3.0 and self.lip_sync is not None:
            try:
                lip_sync_result = self.lip_sync.verify_sync(frames, audio)
                lip_sync_score = self._safe_float(lip_sync_result.get('confidence', 0))
                # Clamp to [0.0, 1.0]
                lip_sync_score = max(0.0, min(1.0, lip_sync_score))
                logger.info(f"Lip-sync liveness score: {lip_sync_score:.2f}")
            except Exception as e:
                logger.info(f"Lip-sync unavailable: {e}")
                lip_sync_result = {
                    'synced': False,
                    'confidence': 0.0,
                    'sync_score': 0.0,
                    'reason': f"Lip-sync error: {str(e)}"
                }
                lip_sync_score = None
        else:
            if audio_duration < 3.0:
                logger.info(f"Lip-sync skipped: audio too short ({audio_duration:.1f}s < 3s)")
            elif faces_detected == 0:
                logger.info("Lip-sync skipped: no face detected")
            else:
                logger.info("Lip-sync unavailable (verifier not initialized)")
            lip_sync_result = {
                'synced': False,
                'confidence': 0.0,
                'sync_score': 0.0,
                'reason': 'Lip sync unavailable'
            }
            lip_sync_score = None
        
        # 5. Age prediction (with error handling)
        try:
            age_result = self._predict_age(frames, audio, sample_rate)
        except Exception as e:
            logger.info(f"Age prediction failed: {e}")
            age_result = {
                'age': 0,
                'face_age': None,
                'voice_age': None,
                'age_group': 'unknown',
                'age_group_confidence': 0.0,
                'is_adult': False,
                'adult_confidence': 0.0,
                'confidence': 0.0
            }
        
        # No demo fallback - captcha must match strictly
        
        # 6. Calculate detailed metrics (using cached face crops)
        face_detection_details = {
            'frames_analyzed': len(frames),
            'faces_detected': len(face_crops),
            'face_quality_score': self._calculate_face_quality_from_crops(face_crops)
        }
        
        # Fix audio duration calculation
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio)
        
        if duration is None:
            if len(audio.shape) > 1:
                num_samples = audio.shape[0]
            else:
                num_samples = len(audio)
            audio_duration = num_samples / sample_rate if sample_rate > 0 else 0
        else:
            audio_duration = duration
        
        voice_analysis_details = {
            'audio_duration': audio_duration,
            'speech_detected': voice_liveness_result.get('speech_detected', True),
            'voice_quality_score': self._calculate_voice_quality(audio, sample_rate)
        }
        
        blink_analysis = {
            'blinks_detected': blink_info.get('blink_count', 0),
            'blink_rate': blink_info.get('blink_rate', 0),
            'blink_pattern_valid': blink_info.get('is_valid', False)
        }
        
        lip_sync_analysis = {
            'sync_score': self._safe_float(lip_sync_score) if lip_sync_score is not None else None,
            'temporal_alignment': self._safe_float(lip_sync_result.get('lag_ms', 0)) if 'lag_ms' in lip_sync_result else 0.0,
            'phoneme_match': self._safe_float(lip_sync_result.get('correlation', 0)) if 'correlation' in lip_sync_result else 0.0
        }
        
        # 7. Combine all liveness scores
        face_live_score = face_liveness_result.get('liveness_score', 0)
        voice_live_score = voice_liveness_result.get('confidence', 0)
        # Use lip_sync_score from above (already clamped and logged)
        
        # Overall liveness determination (more lenient - at least 2 checks should pass)
        liveness_checks = [
            face_liveness_result.get('is_live', False),
            voice_liveness_result.get('verified', False),
            blink_info.get('is_valid', False),
            head_movement.get('motion_detected', False)
        ]
        passed_checks = sum(liveness_checks)
        
        # Require at least 2 out of 4 checks to pass (more lenient for testing)
        is_live = passed_checks >= 2
        
        # 9. Calculate overall confidence with updated weights
        # face_liveness → 0.45, eye_blink → 0.15 (if available), lip_sync → 0.25 (if available), voice_liveness → 0.15 (if available)
        # NOTE: Low eye_blink (0.4) and lip_sync (0.05-0.15) scores are BY DESIGN and CORRECT
        # - Eye blink: 1 blink = 0.4 (40%) is expected for short recordings
        # - Lip-sync: 0.05-0.15 correlation is normal for calm/slow speech
        # These are CONFIDENCE signals only - they NEVER gate success
        # DO NOT inflate, boost, or add fallbacks - low scores are honest measurements
        confidence_components = []
        face_age_val = age_result.get('face_age')
        voice_age_val = age_result.get('voice_age')
        
        # Face liveness (primary, 0.45 weight)
        if face_age_val is not None:
            confidence_components.append(face_live_score * 0.45)
        
        # Eye blink (0.15 weight if available)
        # Accepts low scores (0.4 for 1 blink) - this is CORRECT behavior
        if eye_blink_score is not None and eye_blink_score > 0:
            confidence_components.append(eye_blink_score * 0.15)
        
        # Lip-sync (0.25 weight if available)
        # Accepts low scores (0.05-0.15) - this is NORMAL for real humans
        if lip_sync_score is not None and lip_sync_score > 0:
            confidence_components.append(lip_sync_score * 0.25)
        
        # Voice liveness (0.15 weight ONLY if voice_age available)
        if voice_age_val is not None:
            confidence_components.append(voice_live_score * 0.15)
        
        overall_confidence = self._safe_float(sum(confidence_components) if confidence_components else 0.0)
        # Clamp confidence to [0.0, 1.0] - NO NaN / Inf allowed
        overall_confidence = max(0.0, min(1.0, overall_confidence))
        
        # 9. Success criteria - strict rule
        # Success if: captcha verified AND face_age IS NOT None
        captcha_verified = voice_liveness_result.get('verified', False)
        success = (
            captcha_verified and
            face_age_val is not None
        )
        
        # 10. Complete captcha session
        if success:
            self.captcha_session.complete_session(captcha_id)
        
        # Sanitize all numeric values before returning
        response = {
            'success': success,
            'session_id': session_id,
            'verification_id': verification_id,
            'timestamp': timestamp,
            'liveness': {
                'is_live': is_live,
                'face_liveness_score': self._safe_float(face_live_score),
                'voice_liveness_score': self._safe_float(voice_live_score),
                'lip_sync_score': self._safe_float(lip_sync_score) if lip_sync_score is not None else None,
                'eye_blink_score': self._safe_float(eye_blink_score) if eye_blink_score is not None else None,
                'blink_detected': blink_info.get('blink_count', 0) > 0,
                'blink_count': int(blink_info.get('blink_count', 0)),
                'head_movement_detected': head_movement.get('motion_detected', False),
                'motion_score': self._safe_float(head_movement.get('motion_score', 0)),
                'captcha_verified': voice_liveness_result.get('verified', False),
                'captcha_match_score': self._safe_float(voice_liveness_result.get('match_score', voice_liveness_result.get('confidence', 0))),
                'reasons': self._collect_failure_reasons(
                    face_liveness_result,
                    voice_liveness_result,
                    lip_sync_result,
                    blink_info,
                    head_movement
                )
            },
            'age': {
                'estimated_age': self._safe_float(age_result.get('age', 0)),
                'face_age': self._safe_float(age_result.get('face_age')) if age_result.get('face_age') is not None else None,
                'voice_age': self._safe_float(age_result.get('voice_age')) if age_result.get('voice_age') is not None else None,
                'age_group': age_result.get('age_group', 'unknown'),
                'age_group_confidence': self._safe_float(age_result.get('age_group_confidence', 0)),
                'is_adult': age_result.get('is_adult', False),
                'adult_confidence': self._safe_float(age_result.get('adult_confidence', 0))
            },
            'analysis_details': {
                'face_detection': {
                    'frames_analyzed': int(face_detection_details.get('frames_analyzed', 0)),
                    'faces_detected': int(face_detection_details.get('faces_detected', 0)),
                    'face_quality_score': self._safe_float(face_detection_details.get('face_quality_score', 0))
                },
                'voice_analysis': {
                    'audio_duration': self._safe_float(voice_analysis_details.get('audio_duration', 0)),
                    'speech_detected': voice_analysis_details.get('speech_detected', True),
                    'voice_quality_score': self._safe_float(voice_analysis_details.get('voice_quality_score', 0))
                },
                'blink_analysis': {
                    'blinks_detected': int(blink_analysis.get('blinks_detected', 0)),
                    'blink_rate': self._safe_float(blink_analysis.get('blink_rate', 0)),
                    'blink_pattern_valid': blink_analysis.get('blink_pattern_valid', False)
                },
                'lip_sync_analysis': {
                    'sync_score': self._safe_float(lip_sync_analysis.get('sync_score', 0)),
                    'temporal_alignment': self._safe_float(lip_sync_analysis.get('temporal_alignment', 0)),
                    'phoneme_match': self._safe_float(lip_sync_analysis.get('phoneme_match', 0))
                }
            },
            'overall_confidence': self._safe_float(overall_confidence),
            'message': 'Verification successful' if success else 'Verification failed'
        }
        
        # Sanitize all NumPy types before returning
        return self._sanitize(response)
    
    def _calculate_face_quality_from_crops(self, face_crops: List[np.ndarray]) -> float:
        """Calculate average face quality score from pre-extracted face crops"""
        if not face_crops:
            return 0.0
        
        quality_scores = []
        for face_crop in face_crops:
            try:
                # Validate crop
                if face_crop is None or face_crop.size == 0:
                    continue
                
                # Validate dimensions
                if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                    continue
                
                # Convert to grayscale safely
                if len(face_crop.shape) == 3:
                    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                else:
                    gray = face_crop
                
                # Calculate sharpness using Laplacian variance
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                # Normalize to 0-1 range (typical values: 0-500)
                sharpness_score = min(1.0, laplacian_var / 500.0)
                
                # Calculate brightness (should be well-lit)
                brightness = np.mean(gray) / 255.0
                brightness_score = 1.0 - abs(brightness - 0.5) * 2  # Penalize too dark/bright
                
                # Combined quality score
                quality = (sharpness_score * 0.6 + brightness_score * 0.4)
                quality_scores.append(quality)
            except Exception as e:
                logger.debug(f"Face quality calculation failed for crop: {e}")
                continue
        
        if not quality_scores:
            return 0.0
        try:
            mean_score = np.mean(quality_scores)
            return self._safe_float(mean_score)
        except Exception:
            return 0.0
    
    def _calculate_voice_quality(self, audio: np.ndarray, sample_rate: int) -> float:
        """Calculate voice quality score"""
        if len(audio) == 0:
            return 0.0
        
        # Calculate signal-to-noise ratio (simplified)
        signal_power = np.mean(audio ** 2)
        noise_estimate = np.var(audio - np.mean(audio))
        snr = 10 * np.log10(signal_power / (noise_estimate + 1e-10))
        
        # Normalize SNR to 0-1 (typical range: 0-30 dB)
        snr_score = min(1.0, max(0.0, (snr + 10) / 40))
        
        # Check for clipping
        max_amplitude = np.max(np.abs(audio))
        clipping_score = 1.0 if max_amplitude < 0.95 else 0.7
        
        # Check duration (should be reasonable length)
        duration = len(audio) / sample_rate
        duration_score = 1.0 if 2.0 <= duration <= 10.0 else 0.5
        
        # Combined quality score
        quality = (snr_score * 0.5 + clipping_score * 0.3 + duration_score * 0.2)
        
        return self._safe_float(quality)
    
    def _collect_failure_reasons(
        self,
        face_liveness_result: Dict,
        voice_liveness_result: Dict,
        lip_sync_result: Dict,
        blink_info: Dict,
        head_movement: Dict
    ) -> List[str]:
        """Collect all failure reasons from various checks"""
        reasons = []
        
        # Face liveness reasons
        if face_liveness_result.get('reasons'):
            reasons.extend(face_liveness_result['reasons'])
        
        # Voice liveness reasons
        if not voice_liveness_result.get('verified', False):
            reason = voice_liveness_result.get('reason', 'Voice verification failed')
            if reason:
                reasons.append(reason)
        
        # Lip sync reasons
        if lip_sync_result.get('confidence', 0) < 0.5:
            reason = lip_sync_result.get('reason', 'Lip sync mismatch')
            if reason:
                reasons.append(reason)
        
        # Blink analysis reasons
        if not blink_info.get('is_valid', False):
            if blink_info.get('blink_count', 0) == 0:
                reasons.append('No blinks detected')
            elif blink_info.get('blink_rate', 0) < 0.1:
                reasons.append('Insufficient blink rate')
        
        # Head movement reasons
        if not head_movement.get('motion_detected', False):
            reasons.append('Insufficient head movement detected')
        
        return reasons if reasons else ['All checks passed']
    
    def _create_failed_response(
        self,
        verification_id: str,
        timestamp: str,
        session_id: str,
        reason: str
    ) -> Dict:
        """Create a standardized failed response"""
        response = {
            'success': False,
            'session_id': session_id,
            'verification_id': verification_id,
            'timestamp': timestamp,
            'liveness': {
                'is_live': False,
                'face_liveness_score': 0.0,
                'voice_liveness_score': 0.0,
                'lip_sync_score': 0.0,
                'blink_detected': False,
                'blink_count': 0,
                'head_movement_detected': False,
                'motion_score': 0.0,
                'captcha_verified': False,
                'captcha_match_score': 0.0,
                'reasons': [reason]
            },
            'age': {
                'estimated_age': 0,
                'face_age': None,
                'voice_age': None,
                'age_group': 'unknown',
                'age_group_confidence': 0.0,
                'is_adult': False,
                'adult_confidence': 0.0
            },
            'analysis_details': {
                'face_detection': {
                    'frames_analyzed': 0,
                    'faces_detected': 0,
                    'face_quality_score': 0.0
                },
                'voice_analysis': {
                    'audio_duration': 0,
                    'speech_detected': False,
                    'voice_quality_score': 0.0
                },
                'blink_analysis': {
                    'blinks_detected': 0,
                    'blink_rate': 0,
                    'blink_pattern_valid': False
                },
                'lip_sync_analysis': {
                    'sync_score': 0,
                    'temporal_alignment': 0,
                    'phoneme_match': 0
                }
            },
            'overall_confidence': 0.0,
            'message': reason
        }
        
        # Sanitize all NumPy types before returning
        return self._sanitize(response)
    
    def _calculate_overall_confidence(
        self,
        face_live: float,
        voice_live: float,
        lip_sync: float,
        age_conf: float,
        blink_score: float = 0.0,
        motion_score: float = 0.0
    ) -> float:
        """Calculate overall verification confidence with additional factors"""
        # Enhanced weighted combination
        weights = {
            'face_live': 0.25,
            'voice_live': 0.30,
            'lip_sync': 0.15,
            'age': 0.15,
            'blink': 0.10,
            'motion': 0.05
        }
        
        confidence = (
            weights['face_live'] * face_live +
            weights['voice_live'] * voice_live +
            weights['lip_sync'] * lip_sync +
            weights['age'] * age_conf +
            weights['blink'] * blink_score +
            weights['motion'] * motion_score
        )
        
        return min(1.0, max(0.0, confidence))


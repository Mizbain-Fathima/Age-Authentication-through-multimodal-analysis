"""
Services for Age Authentication System
Business logic layer between API routes and ML models
"""
import torch
import numpy as np
import cv2
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import uuid
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import DEVICE, MODELS_DIR, AGE_GROUPS, AGE_THRESHOLD
from src.liveness.face_liveness import FaceLivenessDetector
from src.liveness.voice_liveness import VoiceLivenessDetector
from src.liveness.lip_sync import LipSyncVerifier
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
        self.lip_sync = LipSyncVerifier()
        
        # Captcha session manager
        self.captcha_session = CaptchaSession()
        
        # Try to load pre-trained models
        self._load_models()
        
        logger.info("AgeAuthenticationService initialized")
    
    def _load_models(self):
        """Load pre-trained models if available"""
        # Try to load face model
        face_checkpoint = MODELS_DIR / "face_age_model_best.pth"
        if face_checkpoint.exists():
            try:
                from src.models.face_model import load_face_model
                self.face_model = load_face_model(str(face_checkpoint))
                logger.info("Loaded face model")
            except Exception as e:
                logger.warning(f"Could not load face model: {e}")
        else:
            # Create default model
            try:
                from src.models.face_model import create_face_model
                self.face_model = create_face_model(pretrained=True)
                logger.info("Created default face model (not trained)")
            except Exception as e:
                logger.warning(f"Could not create face model: {e}")
        
        # Try to load voice model
        voice_checkpoint = MODELS_DIR / "voice_age_model_best.pth"
        if voice_checkpoint.exists():
            try:
                from src.models.voice_model import load_voice_model
                self.voice_model = load_voice_model(str(voice_checkpoint))
                logger.info("Loaded voice model")
            except Exception as e:
                logger.warning(f"Could not load voice model: {e}")
        else:
            # Create default model
            try:
                from src.models.voice_model import create_voice_model
                self.voice_model = create_voice_model()
                logger.info("Created default voice model (not trained)")
            except Exception as e:
                logger.warning(f"Could not create voice model: {e}")
        
        # Try to load fusion model
        fusion_checkpoint = MODELS_DIR / "fusion_model_best.pth"
        if fusion_checkpoint.exists():
            try:
                from src.models.fusion_model import load_fusion_model
                self.fusion_model = load_fusion_model(str(fusion_checkpoint))
                logger.info("Loaded fusion model")
            except Exception as e:
                logger.warning(f"Could not load fusion model: {e}")
    
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
            return {
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
        
        # Face liveness detection
        face_liveness_result = {'is_live': False, 'liveness_score': 0, 'reasons': []}
        if frames and len(frames) > 0:
            face_liveness_result = self.face_liveness.analyze_video_sequence(frames)
        
        # Voice liveness (captcha verification)
        voice_liveness_result = {'verified': False, 'confidence': 0, 'reason': 'No audio'}
        if audio is not None:
            voice_liveness_result = self.voice_liveness.verify_captcha(
                audio, expected_text, sample_rate
            )
        
        # Lip sync verification
        lip_sync_result = {'synced': False, 'confidence': 0, 'reason': 'No data'}
        if frames and len(frames) > 0 and audio is not None:
            lip_sync_result = self.lip_sync.verify_sync(frames, audio)
        
        # Age prediction
        age_result = self._predict_age(frames, audio, sample_rate)
        
        # Combine liveness scores
        face_live_score = face_liveness_result.get('liveness_score', 0)
        voice_live_score = voice_liveness_result.get('confidence', 0)
        lip_sync_score = lip_sync_result.get('confidence', 0)
        
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
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            face_live_score, voice_live_score, lip_sync_score,
            age_result.get('confidence', 0)
        )
        
        # Success criteria
        success = is_live and overall_confidence > 0.5
        
        # Complete captcha session
        if success:
            self.captcha_session.complete_session(captcha_id)
        
        return {
            'success': success,
            'verification_id': verification_id,
            'timestamp': timestamp,
            'liveness': {
                'is_live': is_live,
                'face_liveness_score': face_live_score,
                'voice_liveness_score': voice_live_score,
                'lip_sync_score': lip_sync_score,
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
    
    def _predict_age(self, frames: Optional[List[np.ndarray]], 
                     audio: Optional[np.ndarray], sample_rate: int) -> Dict:
        """Predict age from face and/or voice"""
        face_age = None
        voice_age = None
        age_probs = None
        
        # Face age prediction
        if frames and len(frames) > 0 and self.face_model is not None:
            face_result = self.predict_face_age(frames[-1])  # Use last frame
            face_age = face_result.get('age')
            age_probs = face_result.get('age_group_probs')
        
        # Voice age prediction
        if audio is not None and self.voice_model is not None:
            voice_result = self.predict_voice_age(audio, sample_rate)
            voice_age = voice_result.get('age')
            if age_probs is None:
                age_probs = voice_result.get('age_group_probs')
        
        # Fuse predictions
        if face_age is not None and voice_age is not None:
            # Weighted average (face usually more reliable)
            final_age = 0.6 * face_age + 0.4 * voice_age
        elif face_age is not None:
            final_age = face_age
        elif voice_age is not None:
            final_age = voice_age
        else:
            final_age = 0
        
        # Determine age group
        age_group = self._get_age_group(final_age)
        
        # Is adult
        is_adult = final_age >= AGE_THRESHOLD
        adult_confidence = min(1.0, abs(final_age - AGE_THRESHOLD) / 10 + 0.5)
        
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
            'age': age,
            'age_group': predicted_group,
            'age_group_probs': {g: float(p) for g, p in zip(age_groups, age_group_probs)},
            'is_adult': is_adult_prob > 0.5,
            'is_adult_prob': is_adult_prob
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
            'age': age,
            'age_group': predicted_group,
            'age_group_probs': {g: float(p) for g, p in zip(age_groups, age_group_probs)},
            'is_adult': is_adult_prob > 0.5,
            'is_adult_prob': is_adult_prob
        }
    
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


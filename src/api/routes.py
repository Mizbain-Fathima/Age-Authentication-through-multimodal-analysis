"""
API Routes for Age Authentication System
"""
import io
import os
import base64
import json
import time
import numpy as np
import cv2
import uuid
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from loguru import logger

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))


router = APIRouter()

# =====================
# In-Memory Captcha Storage
# =====================

# Simple in-memory storage for captcha sessions
# Format: {captcha_id: {'sentence': str, 'created_at': float}}
captcha_storage = {}


# =====================
# Pydantic Models
# =====================

class CaptchaResponse(BaseModel):
    captcha_id: str
    sentence: str
    expires_in: int


class VerificationRequest(BaseModel):
    captcha_id: str
    video_data: Optional[str] = None  # Base64 encoded video frames
    audio_data: Optional[str] = None  # Base64 encoded audio


class AgeGroupResult(BaseModel):
    group: str  # child, teen, adult, senior
    confidence: float


class LivenessResult(BaseModel):
    is_live: bool
    face_liveness_score: float
    voice_liveness_score: float
    lip_sync_score: float
    captcha_verified: bool
    reasons: List[str]


class AgeResult(BaseModel):
    estimated_age: float
    face_age: Optional[float]
    voice_age: Optional[float]
    age_group: str
    age_group_confidence: float
    is_adult: bool
    adult_confidence: float


class VerificationResponse(BaseModel):
    success: bool
    verification_id: str
    timestamp: str
    liveness: LivenessResult
    age: AgeResult
    overall_confidence: float
    message: str


class SingleImageRequest(BaseModel):
    image_data: str  # Base64 encoded image


class SingleAudioRequest(BaseModel):
    audio_data: str  # Base64 encoded audio


class ProcessInitRequest(BaseModel):
    complexity: str = "medium"


class ProcessInitResponse(BaseModel):
    session_id: str
    captcha_id: str
    captcha_sentence: str
    expires_in: int
    status: str


class BlinkAnalysis(BaseModel):
    blinks_detected: int
    blink_rate: float
    blink_pattern_valid: bool


class LipSyncAnalysis(BaseModel):
    sync_score: float
    temporal_alignment: float
    phoneme_match: float


class FaceDetectionDetails(BaseModel):
    frames_analyzed: int
    faces_detected: int
    face_quality_score: float


class VoiceAnalysisDetails(BaseModel):
    audio_duration: float
    speech_detected: bool
    voice_quality_score: float


class AnalysisDetails(BaseModel):
    face_detection: FaceDetectionDetails
    voice_analysis: VoiceAnalysisDetails
    blink_analysis: BlinkAnalysis
    lip_sync_analysis: LipSyncAnalysis


class EnhancedLivenessResult(BaseModel):
    is_live: bool
    face_liveness_score: float
    voice_liveness_score: float
    lip_sync_score: float
    blink_detected: bool
    blink_count: int
    head_movement_detected: bool
    motion_score: float
    captcha_verified: bool
    captcha_match_score: float
    reasons: List[str]


class ProcessResponse(BaseModel):
    success: bool
    session_id: str
    verification_id: str
    timestamp: str
    liveness: EnhancedLivenessResult
    age: AgeResult
    analysis_details: AnalysisDetails
    overall_confidence: float
    message: str


# =====================
# API Endpoints
# =====================

@router.get("/status")
async def get_status(request: Request):
    """Return status of loaded models (face, voice, fusion). Use to confirm fusion model is loaded."""
    auth_service = request.app.state.auth_service
    return auth_service.get_model_status()


@router.post("/process")
async def process_authentication(
    request: Request,
    action: str = Form(...),
    video_file: UploadFile = File(None),
    audio_file: UploadFile = File(None),
    captcha_id: str = Form(None),
    sample_rate: int = Form(16000),
    duration: float = Form(None)
):
    """
    Age authentication process with two actions:
    - action=start: Generate and return captcha sentence
    - action=verify: Perform full authentication with video/audio
    """
    auth_service = request.app.state.auth_service
    
    try:
        # Action: Start - Generate captcha
        if action == "start":
            # Generate captcha
            captcha_data = auth_service.generate_captcha('medium')
            captcha_id = captcha_data['captcha_id']
            captcha_sentence = captcha_data['sentence']
            
            # Store in memory
            captcha_storage[captcha_id] = {
                'sentence': captcha_sentence,
                'created_at': time.time()
            }
            
            # Clean up old captchas (older than 15 min); captcha is valid until verify is submitted
            current_time = time.time()
            max_age = 900  # 15 min max age for cleanup only; single-use on verify
            expired_ids = [
                cid for cid, data in captcha_storage.items()
                if current_time - data['created_at'] > max_age
            ]
            for cid in expired_ids:
                del captcha_storage[cid]
            
            logger.info(f"Generated captcha: {captcha_id[:8]}...")
            
            # Captcha stays valid until recording stops and verify is submitted (then deleted)
            return {
                'action': 'start',
                'captcha_id': captcha_id,
                'captcha_sentence': captcha_sentence,
                'expires_in': max_age,
                'message': 'Please read the sentence aloud while recording'
            }
        
        # Action: Verify - Perform authentication
        elif action == "verify":
            # Validate captcha_id
            if not captcha_id or captcha_id not in captcha_storage:
                response = {
                    'success': False,
                    'estimated_age': None,
                    'is_adult': False,
                    'confidence': 0.0,
                    'checks': {
                        'face_detected': False,
                        'face_liveness': 0.0,
                        'blink_detected': False,
                        'head_movement': False,
                        'captcha_verified': False,
                        'voice_liveness': 0.0,
                        'lip_sync': 0.0
                    },
                    'message': 'Invalid or expired captcha. Please start a new verification.'
                }
                return auth_service._sanitize(response)
            
            # Get expected captcha text
            expected_text = captcha_storage[captcha_id]['sentence']
            
            # Decode video
            frames = []
            if video_file:
                video_bytes = await video_file.read()
                frames = _decode_video(video_bytes)
            
            # Decode audio
            audio_array = None
            if audio_file:
                try:
                    audio_bytes = await audio_file.read()
                    if len(audio_bytes) == 0:
                        logger.warning("Empty audio file received")
                    else:
                        audio_array, sr = _decode_audio(audio_bytes)
                        sample_rate = sr
                        
                        # Fail fast: Check audio validity
                        if audio_array is None or len(audio_array) == 0:
                            logger.warning("Audio decoding resulted in empty array")
                        elif len(audio_array) < sample_rate:  # Less than 1 second
                            logger.warning(f"Audio too short: {len(audio_array)} samples < {sample_rate}")
                            response = {
                                'success': False,
                                'estimated_age': None,
                                'is_adult': False,
                                'confidence': 0.0,
                                'checks': {
                                    'face_detected': False,
                                    'face_liveness': 0.0,
                                    'blink_detected': False,
                                    'head_movement': False,
                                    'captcha_verified': False,
                                    'voice_liveness': 0.0,
                                    'lip_sync': 0.0
                                },
                                'message': 'Audio not detected. Please speak clearly for at least 5 seconds.'
                            }
                            return auth_service._sanitize(response)
                except Exception as e:
                    logger.error(f"Audio decoding failed: {e}")
                    response = {
                        'success': False,
                        'estimated_age': None,
                        'is_adult': False,
                        'confidence': 0.0,
                        'checks': {
                            'face_detected': False,
                            'face_liveness': 0.0,
                            'blink_detected': False,
                            'head_movement': False,
                            'captcha_verified': False,
                            'voice_liveness': 0.0,
                            'lip_sync': 0.0
                        },
                        'message': f'Audio decoding failed: {str(e)}. Please ensure microphone is working.'
                    }
                    return auth_service._sanitize(response)
            
            # Validate inputs
            if len(frames) == 0 and audio_array is None:
                response = {
                    'success': False,
                    'estimated_age': None,
                    'is_adult': False,
                    'confidence': 0.0,
                    'checks': {
                        'face_detected': False,
                        'face_liveness': 0.0,
                        'blink_detected': False,
                        'head_movement': False,
                        'captcha_verified': False,
                        'voice_liveness': 0.0,
                        'lip_sync': 0.0
                    },
                    'message': 'No video or audio data provided'
                }
                return auth_service._sanitize(response)
            
            # Perform authentication with captcha
            result = auth_service.process_authentication(
                frames=frames,
                audio=audio_array if audio_array is not None else np.array([]),
                sample_rate=sample_rate,
                duration=duration,
                expected_captcha_text=expected_text,
                captcha_id=captcha_id
            )
            
            # Clean up used captcha
            if captcha_id in captcha_storage:
                del captcha_storage[captcha_id]
            
            logger.info(f"Authentication completed: success={result['success']}")
            
            return result
        
        else:
            response = {
                'success': False,
                'estimated_age': None,
                'is_adult': False,
                'confidence': 0.0,
                'checks': {
                    'face_detected': False,
                    'face_liveness': 0.0,
                    'blink_detected': False,
                    'head_movement': False,
                    'captcha_verified': False,
                    'voice_liveness': 0.0,
                    'lip_sync': 0.0
                },
                'message': f'Invalid action: {action}. Use "start" or "verify".'
            }
            return auth_service._sanitize(response)
    
    except Exception as e:
        logger.error(f"Process authentication error: {e}")
        # Always return HTTP 200 with failure response
        response = {
            'success': False,
            'estimated_age': None,
            'is_adult': False,
            'confidence': 0.0,
            'checks': {
                'face_detected': False,
                'face_liveness': 0.0,
                'blink_detected': False,
                'head_movement': False,
                'captcha_verified': False,
                'voice_liveness': 0.0,
                'lip_sync': 0.0
            },
            'message': f'Verification failed: {str(e)}'
        }
        return auth_service._sanitize(response)


# =====================
# Helper Functions
# =====================

def _decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode image bytes to numpy array"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    return img


def _decode_base64_image(data: str) -> np.ndarray:
    """Decode base64 image to numpy array"""
    # Remove data URL prefix if present
    if 'base64,' in data:
        data = data.split('base64,')[1]
    
    image_bytes = base64.b64decode(data)
    return _decode_image(image_bytes)


def _decode_video(video_bytes: bytes) -> List[np.ndarray]:
    """Decode video bytes to list of frames"""
    # Write to temp file and read with OpenCV
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as f:
        f.write(video_bytes)
        temp_path = f.name
    
    try:
        cap = cv2.VideoCapture(temp_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        if not frames:
            raise ValueError("No frames extracted from video")
        
        return frames
    finally:
        os.unlink(temp_path)


def _decode_base64_frames(data: str) -> List[np.ndarray]:
    """Decode base64 encoded frames (JSON array of base64 images)"""
    import json
    
    try:
        frame_list = json.loads(data)
        frames = []
        for f in frame_list:
            # Handle data URL format (data:image/jpeg;base64,...)
            if isinstance(f, str) and 'base64,' in f:
                f = f.split('base64,')[1]
            frames.append(_decode_base64_image(f))
        return frames
    except json.JSONDecodeError:
        # Single frame - handle data URL format
        if isinstance(data, str) and 'base64,' in data:
            data = data.split('base64,')[1]
        return [_decode_base64_image(data)]
    except Exception as e:
        logger.error(f"Error decoding frames: {e}")
        raise ValueError(f"Failed to decode video frames: {str(e)}")


def _decode_audio(audio_bytes: bytes) -> tuple:
    """Decode audio bytes to numpy array - supports WAV, FLAC, WebM, MP3"""
    import librosa
    import tempfile
    
    # Check if it's a format that soundfile can handle (WAV, FLAC)
    is_wav_or_flac = (
        audio_bytes[:4] == b'RIFF' or  # WAV
        audio_bytes[:4] == b'fLaC' or  # FLAC
        audio_bytes[:4] == b'FORM'     # AIFF
    )
    
    if is_wav_or_flac:
        # Try soundfile first for WAV/FLAC (faster)
        try:
            import soundfile as sf
            audio_io = io.BytesIO(audio_bytes)
            audio, sample_rate = sf.read(audio_io)
            # Ensure mono
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            return audio, sample_rate
        except Exception as e:
            logger.debug(f"soundfile failed, trying librosa: {e}")
    
    # Use librosa for WebM, MP3, and other formats
    # Determine file extension based on content
    suffix = '.webm'
    if audio_bytes[:4] == b'RIFF':
        suffix = '.wav'
    elif audio_bytes[:3] == b'ID3' or audio_bytes[:2] == b'\xff\xfb':
        suffix = '.mp3'
    elif audio_bytes[:4] == b'fLaC':
        suffix = '.flac'
    
    # Write to temp file for librosa (using context manager for better cleanup)
    temp_fd = None
    temp_path = None
    try:
        temp_fd, temp_path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(temp_fd, 'wb') as f:
            f.write(audio_bytes)
        temp_fd = None  # File handle closed by context manager
        
        # Load with librosa (handles WebM, MP3, WAV, etc.)
        audio, sample_rate = librosa.load(temp_path, sr=16000, mono=True)
        
        return audio, sample_rate
    except Exception as e:
        logger.error(f"Failed to decode audio: {e}")
        raise ValueError(f"Unsupported audio format or corrupted file: {str(e)}")
    finally:
        # Ensure cleanup even on errors
        if temp_fd is not None:
            try:
                os.close(temp_fd)
            except Exception:
                pass
        if temp_path is not None:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {temp_path}: {e}")


def _decode_base64_audio(data: str) -> tuple:
    """Decode base64 audio to numpy array"""
    if 'base64,' in data:
        data = data.split('base64,')[1]
    
    audio_bytes = base64.b64decode(data)
    return _decode_audio(audio_bytes)




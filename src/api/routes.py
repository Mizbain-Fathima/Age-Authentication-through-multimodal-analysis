"""
API Routes for Age Authentication System
"""
import io
import base64
import numpy as np
import cv2
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from loguru import logger

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))


router = APIRouter()


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


# =====================
# API Endpoints
# =====================

@router.get("/captcha", response_model=CaptchaResponse)
async def get_captcha(request: Request, complexity: str = "medium"):
    """
    Generate a new captcha for voice liveness verification
    
    Args:
        complexity: 'simple', 'medium', or 'complex'
    
    Returns:
        Captcha sentence and ID
    """
    auth_service = request.app.state.auth_service
    captcha = auth_service.generate_captcha(complexity)
    
    logger.info(f"Generated captcha: {captcha['captcha_id'][:8]}...")
    
    return CaptchaResponse(**captcha)


@router.post("/verify", response_model=VerificationResponse)
async def verify_age(
    request: Request,
    captcha_id: str = Form(...),
    video: UploadFile = File(None),
    audio: UploadFile = File(None),
    video_frames: str = Form(None),  # Base64 encoded frames
    audio_data: str = Form(None)  # Base64 encoded audio
):
    """
    Complete age verification with liveness detection
    
    Accepts either:
    - File uploads (video, audio)
    - Base64 encoded data (video_frames, audio_data)
    
    Returns:
        Complete verification results including age estimation and liveness
    """
    auth_service = request.app.state.auth_service
    
    try:
        # Process video input
        frames = None
        if video:
            video_bytes = await video.read()
            frames = _decode_video(video_bytes)
        elif video_frames:
            frames = _decode_base64_frames(video_frames)
        
        # Process audio input
        audio_array = None
        sample_rate = 16000
        if audio:
            audio_bytes = await audio.read()
            audio_array, sample_rate = _decode_audio(audio_bytes)
        elif audio_data:
            audio_array, sample_rate = _decode_base64_audio(audio_data)
        
        if frames is None and audio_array is None:
            raise HTTPException(status_code=400, detail="No video or audio data provided")
        
        # Perform verification
        result = auth_service.verify(
            captcha_id=captcha_id,
            frames=frames,
            audio=audio_array,
            sample_rate=sample_rate
        )
        
        logger.info(f"Verification completed: {result['verification_id'][:8]}...")
        
        return VerificationResponse(**result)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Verification error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during verification")


@router.post("/predict/face")
async def predict_face_age(
    request: Request,
    image: UploadFile = File(None),
    image_data: str = Form(None)
):
    """
    Predict age from a single face image
    
    Args:
        image: Uploaded image file
        image_data: Base64 encoded image
    
    Returns:
        Age prediction results
    """
    auth_service = request.app.state.auth_service
    
    try:
        # Decode image
        if image:
            image_bytes = await image.read()
            img_array = _decode_image(image_bytes)
        elif image_data:
            img_array = _decode_base64_image(image_data)
        else:
            raise HTTPException(status_code=400, detail="No image provided")
        
        # Predict
        result = auth_service.predict_face_age(img_array)
        
        return result
    
    except Exception as e:
        logger.error(f"Face prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/voice")
async def predict_voice_age(
    request: Request,
    audio: UploadFile = File(None),
    audio_data: str = Form(None)
):
    """
    Predict age from audio
    
    Args:
        audio: Uploaded audio file
        audio_data: Base64 encoded audio
    
    Returns:
        Age prediction results
    """
    auth_service = request.app.state.auth_service
    
    try:
        # Decode audio
        if audio:
            audio_bytes = await audio.read()
            audio_array, sample_rate = _decode_audio(audio_bytes)
        elif audio_data:
            audio_array, sample_rate = _decode_base64_audio(audio_data)
        else:
            raise HTTPException(status_code=400, detail="No audio provided")
        
        # Predict
        result = auth_service.predict_voice_age(audio_array, sample_rate)
        
        return result
    
    except Exception as e:
        logger.error(f"Voice prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/liveness/face")
async def check_face_liveness(
    request: Request,
    video: UploadFile = File(None),
    video_frames: str = Form(None)
):
    """
    Check face liveness from video frames
    
    Args:
        video: Uploaded video file
        video_frames: Base64 encoded frames
    
    Returns:
        Face liveness detection results
    """
    auth_service = request.app.state.auth_service
    
    try:
        # Decode video
        if video:
            video_bytes = await video.read()
            frames = _decode_video(video_bytes)
        elif video_frames:
            frames = _decode_base64_frames(video_frames)
        else:
            raise HTTPException(status_code=400, detail="No video data provided")
        
        # Check liveness
        result = auth_service.check_face_liveness(frames)
        
        return result
    
    except Exception as e:
        logger.error(f"Face liveness error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/liveness/voice")
async def check_voice_liveness(
    request: Request,
    captcha_id: str = Form(...),
    audio: UploadFile = File(None),
    audio_data: str = Form(None)
):
    """
    Check voice liveness via captcha verification
    
    Args:
        captcha_id: The captcha ID to verify against
        audio: Uploaded audio file
        audio_data: Base64 encoded audio
    
    Returns:
        Voice liveness/captcha verification results
    """
    auth_service = request.app.state.auth_service
    
    try:
        # Decode audio
        if audio:
            audio_bytes = await audio.read()
            audio_array, sample_rate = _decode_audio(audio_bytes)
        elif audio_data:
            audio_array, sample_rate = _decode_base64_audio(audio_data)
        else:
            raise HTTPException(status_code=400, detail="No audio provided")
        
        # Check liveness
        result = auth_service.check_voice_liveness(captcha_id, audio_array, sample_rate)
        
        return result
    
    except Exception as e:
        logger.error(f"Voice liveness error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/status")
async def get_model_status(request: Request):
    """Get status of loaded models"""
    auth_service = request.app.state.auth_service
    return auth_service.get_model_status()


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
        frames = [_decode_base64_image(f) for f in frame_list]
        return frames
    except json.JSONDecodeError:
        # Single frame
        return [_decode_base64_image(data)]


def _decode_audio(audio_bytes: bytes) -> tuple:
    """Decode audio bytes to numpy array"""
    import soundfile as sf
    
    audio_io = io.BytesIO(audio_bytes)
    
    try:
        audio, sample_rate = sf.read(audio_io)
    except Exception:
        # Try with librosa for more formats
        import librosa
        audio_io.seek(0)
        
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name
        
        try:
            audio, sample_rate = librosa.load(temp_path, sr=16000)
        finally:
            os.unlink(temp_path)
    
    # Ensure mono
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    return audio, sample_rate


def _decode_base64_audio(data: str) -> tuple:
    """Decode base64 audio to numpy array"""
    if 'base64,' in data:
        data = data.split('base64,')[1]
    
    audio_bytes = base64.b64decode(data)
    return _decode_audio(audio_bytes)




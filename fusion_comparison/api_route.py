"""
Fusion comparison API — no changes to existing routes or services.
- GET /compare: serve comparison page
- POST /api/process-with-fusion: same as /api/process verify, plus fusion age
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import cv2
import torch
from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import FileResponse
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODELS_DIR, FACE_IMAGE_SIZE, SAMPLE_RATE, DEVICE
from src.api.routes import captcha_storage
from src.api.routes import _decode_video, _decode_audio
from src.liveness.face_liveness import FaceLivenessDetector
from src.data.audio_dataset import LiveAudioProcessor
from torchvision import transforms


router = APIRouter()

# Cached fusion model (lazy load)
_fusion_model = None
_face_detector = None


def _get_fusion_model():
    global _fusion_model
    if _fusion_model is not None:
        return _fusion_model
    fusion_ckpt = MODELS_DIR / "fusion_age_model_best.pth"
    if not fusion_ckpt.exists():
        return None
    from src.models.fusion_model import load_fusion_model
    _fusion_model = load_fusion_model(str(fusion_ckpt), device=DEVICE)
    return _fusion_model


def _get_face_detector():
    global _face_detector
    if _face_detector is None:
        _face_detector = FaceLivenessDetector()
    return _face_detector


def _face_tensor_from_frames(frames, device):
    if not frames:
        return None
    detector = _get_face_detector()
    frame = frames[-1]
    face_crop = detector.get_face_crop(frame)
    if face_crop is None:
        return None
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(FACE_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    return transform(face_rgb).unsqueeze(0).to(device)


def _voice_tensor_from_audio(audio_array, sample_rate, device):
    if audio_array is None or len(audio_array) == 0:
        return None
    processor = LiveAudioProcessor()
    features = processor.process_audio_bytes(audio_array, sample_rate)
    return features.to(device)


@router.get("/compare")
async def serve_compare_page():
    """Serve the fusion comparison page."""
    compare_path = Path(__file__).parent / "compare.html"
    if not compare_path.exists():
        return {"error": "compare.html not found"}, 404
    return FileResponse(str(compare_path))


@router.post("/api/process-with-fusion")
async def process_with_fusion(
    request: Request,
    action: str = Form(...),
    video_file: UploadFile = File(None),
    audio_file: UploadFile = File(None),
    captcha_id: str = Form(None),
    sample_rate: int = Form(16000),
    duration: float = Form(None),
):
    """
    Same as /api/process verify, plus fusion model age.
    For action=start use existing POST /api/process. This endpoint only handles action=verify.
    """
    if action != "verify":
        return {
            "success": False,
            "estimated_age": None,
            "fusion_age": None,
            "message": "Use action=verify. For captcha use POST /api/process with action=start.",
        }

    auth_service = request.app.state.auth_service
    if not captcha_id or captcha_id not in captcha_storage:
        out = {
            "success": False,
            "estimated_age": None,
            "fusion_age": None,
            "is_adult": False,
            "confidence": 0.0,
            "checks": {},
            "message": "Invalid or expired captcha. Please start a new verification.",
        }
        return auth_service._sanitize(out)

    expected_text = captcha_storage[captcha_id]["sentence"]

    frames = []
    if video_file:
        video_bytes = await video_file.read()
        try:
            frames = _decode_video(video_bytes)
        except Exception:
            frames = []

    audio_array = None
    sr = sample_rate
    if audio_file:
        try:
            audio_bytes = await audio_file.read()
            if len(audio_bytes) > 0:
                audio_array, sr = _decode_audio(audio_bytes)
        except Exception:
            audio_array = None

    if len(frames) == 0 and (audio_array is None or len(audio_array) == 0):
        out = {
            "success": False,
            "estimated_age": None,
            "fusion_age": None,
            "is_adult": False,
            "confidence": 0.0,
            "checks": {},
            "message": "No video or audio data provided",
        }
        return auth_service._sanitize(out)

    result = auth_service.process_authentication(
        frames=frames,
        audio=audio_array if audio_array is not None else np.array([]),
        sample_rate=sr,
        duration=duration,
        expected_captcha_text=expected_text,
        captcha_id=captcha_id,
    )

    if captcha_id in captcha_storage:
        del captcha_storage[captcha_id]

    fusion_age = None
    fusion_age_group = None
    fusion_is_adult = None
    fusion_model = _get_fusion_model()
    if fusion_model is None:
        logger.debug("Fusion comparison: fusion model not loaded (check models/fusion_age_model_best.pth)")
    elif len(frames) == 0 or audio_array is None or len(audio_array) <= sr:
        logger.debug("Fusion comparison: skipped (need frames and audio > 1s)")
    else:
        face_tensor = _face_tensor_from_frames(frames, DEVICE)
        voice_tensor = _voice_tensor_from_audio(audio_array, sr, DEVICE)
        if face_tensor is None or voice_tensor is None:
            logger.debug("Fusion comparison: skipped (no face crop or voice tensor)")
        else:
            try:
                fusion_model.eval()
                with torch.no_grad():
                    out = fusion_model(face_tensor, voice_tensor)
                fusion_age = float(out["age"].item())
                fusion_age = max(0, min(100, fusion_age))
                idx = out["age_group"].item()
                fusion_age_group = ["child", "teen", "adult", "senior"][idx]
                fusion_is_adult = bool((out["is_adult"] > 0.5).item())
                logger.info(f"Fusion comparison: fusion_age={fusion_age:.1f}")
            except Exception as e:
                logger.warning(f"Fusion comparison failed: {e}")

    result["fusion_age"] = fusion_age
    result["fusion_age_group"] = fusion_age_group
    result["fusion_is_adult"] = fusion_is_adult
    if fusion_age is not None and result.get("estimated_age") is not None:
        result["age_difference"] = round(fusion_age - result["estimated_age"], 1)

    return auth_service._sanitize(result)

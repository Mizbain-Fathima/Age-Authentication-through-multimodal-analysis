"""
Compare age prediction: existing pipeline (0.6*face + 0.4*voice) vs fusion model.

Usage (run from project root):
  python fusion_comparison/compare_age.py --video path/to/video.mp4 --audio path/to/audio.wav

Requires: face and voice model checkpoints for "existing"; fusion checkpoint for "fusion".
No changes to any existing project files; only imports from src.
"""

import sys
import argparse
from pathlib import Path

# Project root so we can import src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
from torchvision import transforms

from src.config import MODELS_DIR, FACE_IMAGE_SIZE, SAMPLE_RATE, DEVICE
from src.liveness.face_liveness import FaceLivenessDetector
from src.data.audio_dataset import LiveAudioProcessor


def load_models():
    """Load face, voice, and fusion models from checkpoints (existing code)."""
    face_model = None
    voice_model = None
    fusion_model = None

    face_ckpt = MODELS_DIR / "face_age_model_best.pth"
    if face_ckpt.exists():
        from src.models.face_model import load_face_model
        face_model = load_face_model(str(face_ckpt), device=DEVICE)
    else:
        print(f"Warning: {face_ckpt} not found. Existing pipeline will skip face.")

    voice_ckpt = MODELS_DIR / "voice_age_model_best.pth"
    if voice_ckpt.exists():
        from src.models.voice_model import load_voice_model
        voice_model = load_voice_model(str(voice_ckpt), device=DEVICE)
    else:
        print(f"Warning: {voice_ckpt} not found. Existing pipeline will skip voice.")

    fusion_ckpt = MODELS_DIR / "fusion_age_model_best.pth"
    if fusion_ckpt.exists():
        from src.models.fusion_model import load_fusion_model
        fusion_model = load_fusion_model(str(fusion_ckpt), device=DEVICE)
    else:
        print(f"Warning: {fusion_ckpt} not found. Fusion comparison will be skipped.")

    return face_model, voice_model, fusion_model


def get_face_tensor_from_frame(frame_bgr: np.ndarray, face_detector, device) -> torch.Tensor:
    """Crop face from frame and return preprocessed tensor (1, 3, 160, 160). Same as services."""
    face_crop = face_detector.get_face_crop(frame_bgr)
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


def get_voice_tensor(audio: np.ndarray, sample_rate: int, device) -> torch.Tensor:
    """Return voice feature tensor (1, 2, 128, T). Same as services."""
    processor = LiveAudioProcessor()
    features = processor.process_audio_bytes(audio, sample_rate)
    return features.to(device)


def extract_frame_from_video(video_path: str) -> np.ndarray:
    """Read video and return the last frame (BGR)."""
    cap = cv2.VideoCapture(video_path)
    frame = None
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frame = f
    cap.release()
    return frame


def load_audio(audio_path: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio as mono at sample_rate."""
    try:
        import soundfile as sf
        y, sr = sf.read(audio_path)
    except Exception:
        import librosa
        y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
        return y
    if sr != sample_rate:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=sample_rate)
    if len(y.shape) > 1:
        y = y.mean(axis=1)
    return y


def main():
    parser = argparse.ArgumentParser(
        description="Compare existing (weighted) age vs fusion model age. No changes to existing code."
    )
    parser.add_argument("--video", type=str, required=True, help="Path to video file (e.g. .mp4)")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file (e.g. .wav)")
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE, help="Audio sample rate")
    args = parser.parse_args()

    video_path = Path(args.video)
    audio_path = Path(args.audio)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)
    if not audio_path.exists():
        print(f"Error: Audio not found: {audio_path}")
        sys.exit(1)

    print("Loading models...")
    face_model, voice_model, fusion_model = load_models()
    if face_model is None and voice_model is None:
        print("Error: Need at least face or voice model for existing pipeline.")
        sys.exit(1)

    face_detector = FaceLivenessDetector()

    print("Loading video and audio...")
    frame = extract_frame_from_video(str(video_path))
    if frame is None:
        print("Error: No frames in video.")
        sys.exit(1)
    audio = load_audio(str(audio_path), args.sample_rate)

    face_tensor = get_face_tensor_from_frame(frame, face_detector, DEVICE)
    if face_tensor is None:
        print("Error: No face detected in video.")
        sys.exit(1)
    voice_tensor = get_voice_tensor(audio, args.sample_rate, DEVICE)

    # Existing pipeline: 0.6*face + 0.4*voice (same as process_authentication)
    face_age = None
    voice_age = None
    if face_model is not None:
        face_model.eval()
        with torch.no_grad():
            out = face_model(face_tensor)
        face_age = out["age"].item()
    if voice_model is not None:
        voice_model.eval()
        with torch.no_grad():
            out = voice_model(voice_tensor)
        voice_age = out["age"].item()

    if face_age is not None and voice_age is not None:
        existing_age = 0.6 * face_age + 0.4 * voice_age
        existing_label = "0.6*face + 0.4*voice"
    elif face_age is not None:
        existing_age = face_age
        existing_label = "face only"
    elif voice_age is not None:
        existing_age = voice_age
        existing_label = "voice only"
    else:
        existing_age = None
        existing_label = "N/A"

    fusion_age = None
    if fusion_model is not None:
        fusion_model.eval()
        with torch.no_grad():
            out = fusion_model(face_tensor, voice_tensor)
        fusion_age = out["age"].item()

    # Report
    print("\n" + "=" * 60)
    print("AGE COMPARISON (existing vs fusion)")
    print("=" * 60)
    print(f"  Existing pipeline ({existing_label}): {existing_age:.1f} years" if existing_age is not None else "  Existing: N/A")
    print(f"  Fusion model:                      {fusion_age:.1f} years" if fusion_age is not None else "  Fusion: N/A")
    if existing_age is not None and fusion_age is not None:
        diff = fusion_age - existing_age
        print(f"  Difference (fusion - existing):     {diff:+.1f} years")
    print("=" * 60)
    if face_age is not None:
        print(f"  (Face model alone: {face_age:.1f})")
    if voice_age is not None:
        print(f"  (Voice model alone: {voice_age:.1f})")
    print()


if __name__ == "__main__":
    main()

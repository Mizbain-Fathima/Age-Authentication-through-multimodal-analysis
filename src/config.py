"""
Configuration settings for the Age Authentication System
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent.absolute()
IMAGE_DATA_DIR = BASE_DIR / "image-data" / "UTKFace"
AUDIO_DATA_DIR = BASE_DIR / "audio-data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Image settings
IMAGE_SIZE = (200, 200)
FACE_IMAGE_SIZE = (160, 160)

# Audio settings
SAMPLE_RATE = 16000
N_MFCC = 40
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
MAX_AUDIO_LENGTH = 10  # seconds

# Age group definitions
AGE_GROUPS = {
    'child': (0, 12),
    'teen': (13, 17),
    'adult': (18, 59),
    'senior': (60, 120)
}

AGE_THRESHOLD = 18  # For binary classification (18+)

# Model settings
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Face model settings
FACE_BACKBONE = 'efficientnet_b0'
FACE_DROPOUT = 0.3

# Voice model settings
VOICE_DROPOUT = 0.3

# Fusion model settings
FUSION_HIDDEN_DIM = 256
FUSION_DROPOUT = 0.4

# Liveness detection settings
MIN_FACE_FRAMES = 10
BLINK_THRESHOLD = 0.2
MOTION_THRESHOLD = 5.0
LIP_SYNC_THRESHOLD = 0.6
CAPTCHA_MATCH_THRESHOLD = 0.7

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB

# Audio age mapping (Common Voice categories to numeric)
AUDIO_AGE_MAP = {
    'teens': 15,      # < 19, center at 15
    'twenties': 24,   # 19-29, center at 24
    'thirties': 34,   # 30-39, center at 34
    'fourties': 44,   # 40-49, center at 44
    'fifties': 54,    # 50-59, center at 54
    'sixties': 64,    # 60-69, center at 64
    'seventies': 74,  # 70-79, center at 74
    'eighties': 84,   # 80-89, center at 84
    'nineties': 92    # > 89, approximate at 92
}

# Captcha sentences for liveness verification
CAPTCHA_SENTENCES = [
    "The quick brown fox jumps over the lazy dog",
    "Please verify my identity using this phrase",
    "I am a real person reading this text aloud",
    "Authentication requires speaking clearly",
    "My voice confirms my presence today",
    "Security starts with voice verification",
    "This sentence proves I am present",
    "Speaking naturally helps verify identity",
    "Clear speech enables secure access",
    "Voice recognition ensures safety online",
    "Random words create unique authentication",
    "Technology advances through innovation",
    "Digital security protects personal data",
    "Biometric systems enhance user safety",
    "Modern authentication uses multiple factors",
    "Artificial intelligence powers this system",
    "Speaking aloud confirms my identity",
    "Voice patterns are unique to each person",
    "Face and voice together ensure accuracy",
    "Multimodal systems provide better security",
    "Please speak this sentence clearly now",
    "Verification systems protect your account",
    "Real time processing ensures quick access",
    "Neural networks analyze speech patterns",
    "Deep learning enables age estimation"
]

# Device configuration
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


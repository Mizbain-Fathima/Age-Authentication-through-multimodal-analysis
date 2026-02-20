"""
Configuration settings for the Age Authentication System
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent.absolute()
IMAGE_DATA_DIR = BASE_DIR / "image-data" / "UTKFace"
AUDIO_DATA_DIR = BASE_DIR / "audio-data"
KIDS_AUDIO_DATA_DIR = BASE_DIR / "kids-audio-data" / "output"
MODELS_DIR = BASE_DIR / "models"
FUSION_MODELS_DIR = MODELS_DIR / "fusion"  # Fusion age model checkpoints (separate from multimodal_age_model_best)
LOGS_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
FUSION_MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Reproducibility (for baseline training scripts)
RANDOM_SEED = 42

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
ADULT_THRESHOLD = 18.0  # Configurable threshold for inference (e.g. 17.8 for calibration)

# Model settings
BATCH_SIZE = 8
EPOCHS = 30
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
# Use fusion model for age estimation (False = use 0.6*face + 0.4*voice; recommended: faster, more stable, better accuracy in practice)
USE_FUSION_FOR_AGE = False

# Liveness detection settings
MIN_FACE_FRAMES = 10
BLINK_THRESHOLD = 0.2
MOTION_THRESHOLD = 5.0
LIP_SYNC_THRESHOLD = 0.6
CAPTCHA_MATCH_THRESHOLD = 0.7
FRAME_SUBSAMPLE_RATE = 3

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

# Kids audio dataset age mapping
# File format: {Gender}{SpeakerID}_{SessionID}_{UtteranceID}.wav
# Assuming children ages 5-12, we estimate based on speaker ID range
KIDS_AGE_RANGE = (5, 12)  # Min and max age for kids dataset

# Captcha sentences for liveness verification
CAPTCHA_SENTENCES = [
    "clear speech enables secure access",
    "the fresh flower runs under the cat",
    "safe systems require human presence",
    "the bright star shines over the house",
    "quick birds fly through the garden",
    "warm books sit near the tree",
    "gentle dogs walk along the river",
    "smart cats jump over the bridge",
    "calm clouds move across the sky",
    "bold words create strong messages",
    "kind people help each other",
    "the happy dog runs through the park",
    "bright flowers grow near the window",
    "fresh water flows under the bridge",
    "clear voices speak with confidence",
    "safe places protect all people",
    "the gentle cat walks beside the tree",
    "warm sunlight shines over the garden",
    "quick birds fly through clear skies",
    "strong systems help protect data"
]

# Device configuration
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


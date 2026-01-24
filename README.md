# 🔐 Multimodal Age Authentication System

A production-ready deep learning system for age verification using **live face video** and **live audio** with **captcha-based liveness detection**.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🎯 Core Features

### Multimodal Age Prediction
- **Face Age Prediction**: EfficientNet-based CNN with transfer learning
- **Voice Age Prediction**: MFCC + Mel Spectrogram features with CNN-LSTM
- **Fusion Network**: Cross-modal attention for robust predictions

### Liveness Detection
- **Face Liveness**: Eye blink detection, head motion, texture analysis
- **Voice Liveness**: Captcha verification via speech-to-text
- **Lip Sync**: Correlation between lip movement and audio

### Output
- Estimated age (continuous regression)
- Age group classification (child / teen / adult / senior)
- Binary 18+ verification with confidence scores

## 📁 Project Structure

```
age-authentication/
├── src/
│   ├── api/                  # FastAPI REST API
│   │   ├── main.py          # Application entry point
│   │   ├── routes.py        # API endpoints
│   │   └── services.py      # Business logic
│   │
│   ├── data/                 # Data handling
│   │   ├── data_analysis.py # Dataset analysis & statistics
│   │   ├── face_dataset.py  # UTKFace PyTorch dataset
│   │   └── audio_dataset.py # Common Voice PyTorch dataset
│   │
│   ├── models/               # Deep learning models
│   │   ├── face_model.py    # Face age CNN (EfficientNet)
│   │   ├── voice_model.py   # Voice age CNN-LSTM
│   │   └── fusion_model.py  # Multimodal fusion network
│   │
│   ├── liveness/             # Liveness detection
│   │   ├── face_liveness.py # Face anti-spoofing
│   │   ├── voice_liveness.py# Speech verification
│   │   ├── lip_sync.py      # Audio-visual sync
│   │   └── captcha.py       # Captcha generation
│   │
│   ├── training/             # Model training
│   │   └── trainer.py       # Training loops
│   │
│   └── config.py             # Configuration settings
│
├── frontend/                 # Web UI
│   ├── index.html
│   ├── styles.css
│   └── app.js
│
├── image-data/               # UTKFace dataset
│   └── UTKFace/
│
├── audio-data/               # Common Voice dataset
│   ├── cv-other-train/
│   ├── cv-other-train.csv
│   └── ...
│
├── models/                   # Saved model checkpoints
├── logs/                     # Training logs
├── requirements.txt
├── run.py                    # Main entry point
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Data Analysis

```bash
python run.py analyze
```

This will:
- Analyze UTKFace and Common Voice datasets
- Show age distribution statistics
- Display class imbalance information
- Generate visualization plots

### 3. Train Models

```bash
# Train face age model
python run.py train face

# Train voice age model
python run.py train voice

# Train Fusion model
python run.py train fusion
```

### 4. Start the API Server

```bash
python run.py server
```

The server will start at `http://localhost:8000`

- **Web UI**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 🔌 API Endpoints

### Get Captcha
```http
GET /api/captcha?complexity=medium
```
Returns a random sentence for the user to read aloud.

### Verify Age
```http
POST /api/verify
Content-Type: multipart/form-data

captcha_id: string
video: file (optional)
audio: file (optional)
video_frames: string (base64 JSON array, optional)
audio_data: string (base64, optional)
```

### Predict Face Age
```http
POST /api/predict/face
Content-Type: multipart/form-data

image: file
```

### Predict Voice Age
```http
POST /api/predict/voice
Content-Type: multipart/form-data

audio: file
```

### Check Face Liveness
```http
POST /api/liveness/face
Content-Type: multipart/form-data

video: file
```

### Check Voice Liveness
```http
POST /api/liveness/voice
Content-Type: multipart/form-data

captcha_id: string
audio: file
```

## 📊 Datasets

### UTKFace (Image Data)
- **Location**: `image-data/UTKFace/`
- **Format**: `[age]_[gender]_[race]_[timestamp].jpg`
- **Samples**: ~21,500 face images
- **Age Range**: 0-116 years

### Mozilla Common Voice (Audio Data)
- **Location**: `audio-data/`
- **Format**: MP3 audio with CSV metadata
- **Age Categories**: teens, twenties, thirties, ... nineties
- **Samples**: ~145,000+ with age labels

## 🧠 Model Architecture

### Face Model (EfficientNet-B0)
```
EfficientNet-B0 Backbone (pretrained)
    ↓
Feature Processor (512 → 256)
    ↓
┌─────────────────┬─────────────────┬─────────────────┐
│   Age Head      │  Age Group Head │   Adult Head    │
│   (regression)  │  (4-class)      │   (binary)      │
└─────────────────┴─────────────────┴─────────────────┘
```

### Voice Model (CNN-LSTM)
```
MFCC + Mel Spectrogram Input
    ↓
CNN Encoder (Conv → Pool → Residual)
    ↓
Bidirectional LSTM (temporal modeling)
    ↓
Self-Attention
    ↓
┌─────────────────┬─────────────────┬─────────────────┐
│   Age Head      │  Age Group Head │   Adult Head    │
│   (regression)  │  (4-class)      │   (binary)      │
└─────────────────┴─────────────────┴─────────────────┘
```

### Fusion Model
```
Face Features ────→ Cross-Modal Attention ←──── Voice Features
                           ↓
                    Gated Fusion
                           ↓
                   Fusion Processor
                           ↓
          ┌────────────┬────────────┬────────────┐
          │  Age Head  │ Group Head │ Adult Head │
          └────────────┴────────────┴────────────┘
```

## 🔒 Liveness Detection

### Face Liveness Signals
1. **Eye Blink Detection**: Eye Aspect Ratio (EAR) tracking
2. **Head Motion**: Pose estimation and movement analysis
3. **Texture Analysis**: Laplacian variance and frequency analysis
4. **Temporal Consistency**: Micro-movement patterns

### Voice Liveness Signals
1. **Captcha Verification**: Speech-to-text matching
2. **Voice Characteristics**: Pitch, tempo, spectral features
3. **Naturalness Score**: Combined audio quality metrics

### Lip Sync Verification
1. **Lip Movement Extraction**: Mouth aspect ratio tracking
2. **Audio Energy Correlation**: Cross-correlation analysis
3. **Temporal Alignment**: Lag detection and compensation

## 📈 Training Details

### Data Preprocessing
- **Images**: Resize to 224×224, normalize, augmentation
- **Audio**: 16kHz, MFCC (40 coefficients), Mel Spectrogram (128 bands)
- **Stratified Splits**: 70% train, 15% validation, 15% test

### Training Configuration
- **Optimizer**: AdamW with weight decay
- **Scheduler**: Cosine Annealing
- **Loss**: Multi-task (MAE + CrossEntropy + BCE)
- **Mixed Precision**: Enabled for GPU training

### Class Balancing
- Stratified sampling by age group
- Computed class weights for loss function

## 🎨 Frontend Features

- Modern dark theme with neon accents
- Real-time webcam preview
- Audio waveform visualization
- Step-by-step verification flow
- Animated processing indicators
- Detailed result breakdown

## 🛠️ Configuration

Edit `src/config.py` to customize:

```python
# Model settings
FACE_BACKBONE = 'efficientnet_b0'
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4

# Liveness thresholds
BLINK_THRESHOLD = 0.2
CAPTCHA_MATCH_THRESHOLD = 0.7
LIP_SYNC_THRESHOLD = 0.6

# Age groups
AGE_GROUPS = {
    'child': (0, 12),
    'teen': (13, 17),
    'adult': (18, 59),
    'senior': (60, 120)
}

AGE_THRESHOLD = 18  # For 18+ verification
```

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Support

For issues and questions, please open a GitHub issue.

---

**Built with ❤️ using PyTorch, FastAPI, and MediaPipe**


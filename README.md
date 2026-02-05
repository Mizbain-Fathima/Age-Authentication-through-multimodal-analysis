# 🔐 Multimodal Age Authentication System

A production-ready deep learning system for age verification using **live face video** and **live audio** with **captcha-based liveness detection**.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 Features

### Multimodal Age Prediction
- **Face Age**: EfficientNet-based CNN with transfer learning (ImageNet pretrained)
- **Voice Age**: MFCC + Mel spectrogram features with lightweight CNN (no LSTM)
- **Fusion**: Optional cross-modal attention and gated fusion over face + voice

### Liveness Detection
- **Face**: Blink detection, head motion, texture analysis
- **Voice**: Captcha verification via speech-to-text (Whisper)
- **Eye Blink**: MediaPipe-based blink detection
- **Lip Sync**: Correlation between lip movement and audio

### Outputs
- **Estimated age** (continuous, apparent age)
- **Age group**: child / teen / adult / senior
- **18+ verification** with confidence scores and liveness breakdown

---

## 📁 Project Structure

```
age-authentication/
├── src/
│   ├── api/                  # FastAPI REST API
│   │   ├── main.py           # App entry, static files, CORS
│   │   ├── routes.py         # Endpoints
│   │   └── services.py       # Business logic, age/liveness
│   ├── data/                 # Datasets & feature extraction
│   │   ├── data_analysis.py  # UTKFace & Common Voice analysis, splits
│   │   ├── face_dataset.py   # UTKFace Dataset, LiveAudioProcessor
│   │   ├── audio_dataset.py  # Common Voice Dataset, MFCC/Mel
│   │   └── fusion_dataset.py # Age-matched face+audio pairing
│   ├── models/               # DL models
│   │   ├── face_model.py     # FaceAgeModel (EfficientNet)
│   │   ├── voice_model.py   # VoiceAgeModel (CNN)
│   │   └── fusion_model.py  # MultimodalFusionModel
│   ├── liveness/             # Liveness components
│   │   ├── face_liveness.py  # Face crop, liveness
│   │   ├── voice_liveness.py# Whisper STT, captcha verify
│   │   ├── lip_sync.py       # Lip-sync verification
│   │   ├── eye_blink.py      # Blink detection
│   │   └── captcha.py        # Captcha generation/session
│   ├── training/             # Training scripts
│   │   └── trainer.py        # FaceTrainer, VoiceTrainer, FusionTrainer
│   └── config.py             # Paths, hyperparams, age groups
├── frontend/                 # Web UI
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── image-data/UTKFace/        # Face images (optional for training)
├── audio-data/                # Common Voice (optional for training)
├── models/                    # Saved checkpoints (face_age_model_best.pth, etc.)
├── logs/                      # Training logs
├── run.py                     # CLI entry point
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Run

### 1. Environment

```bash
# Create and activate venv
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data (optional, for training)

- **Face**: Place UTKFace images under `image-data/UTKFace/` (filename format: `[age]_[gender]_[race]_[date].jpg`).
- **Audio**: Place Mozilla Common Voice data under `audio-data/` (CSVs + audio files).

To inspect datasets and splits:

```bash
python run.py analyze
```

### 3. Train models (optional)

Pre-trained checkpoints can be placed in `models/`. To train from scratch or resume:

```bash
# Face model (EfficientNet backbone)
python run.py train face
python run.py train face --resume   # resume from checkpoint

# Voice model (MFCC + Mel CNN)
python run.py train voice
python run.py train voice --resume

# Fusion (requires face + voice checkpoints; encoders usually frozen)
python run.py train fusion
```

Checkpoints are saved under `models/` (e.g. `face_age_model_best.pth`, `voice_age_model_best.pth`, `fusion_age_model_best.pth`).

### 4. Start the server

```bash
python run.py server
```

- **Web app**: http://localhost:8000  
- **API docs (Swagger)**: http://localhost:8000/docs  
- **Health**: http://localhost:8000/health  

Default: `0.0.0.0:8000` (config in `src/config.py`: `API_HOST`, `API_PORT`).

---

## 🔄 Workflow

1. **User** opens the web UI, starts verification.
2. **Captcha**: Client requests a sentence (`GET /api/captcha`), user reads it aloud while on camera.
3. **Record**: Frontend captures video frames and audio, then sends them in one request (`POST /api/process`).
4. **Backend**:
   - Validates captcha (voice liveness).
   - Runs face liveness, eye blink, lip-sync.
   - Predicts face age (and voice age if audio is valid).
   - Fuses age (e.g. 0.6×face + 0.4×voice when both available), computes 18+ and confidence.
5. **Response**: Success/failure, estimated age, age group, confidence, and liveness breakdown.

---

## 🔌 Main API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serve frontend (index.html) |
| GET | `/health` | Health check |
| GET | `/api/captcha?complexity=medium` | Get captcha sentence and session id |
| POST | `/api/process` | **Full verification**: video frames + audio; returns age, liveness, success |

See **Swagger** at http://localhost:8000/docs for request/response schemas.

---

## 📊 Datasets (for training)

| Dataset | Location | Purpose |
|--------|----------|--------|
| **UTKFace** | `image-data/UTKFace/` | Face age; filename encodes age, gender, race |
| **Mozilla Common Voice** | `audio-data/` | Voice age; CSV + audio; age categories mapped to numeric age |
| **Kids audio** (optional) | `kids-audio-data/output/` | Extra child voice samples |

---

## 🧠 Models (high level)

- **Face**: EfficientNet-B0 backbone → feature MLP → age (regression), age_group (4-class), is_adult (binary).
- **Voice**: Input (B, 2, 128, T) = MFCC-padded + Mel → CNN (depthwise separable + SE + frequency attention) → global pool → MLP → same three heads.
- **Fusion**: Face + voice encoders (frozen or not) → projection → cross-modal attention → gated fusion → MLP → same three heads.

Details (layers, losses, data pipeline) are in **INTERNAL.md**.

---

## 🛠️ Configuration

Edit `src/config.py` for:

- Paths: `IMAGE_DATA_DIR`, `AUDIO_DATA_DIR`, `MODELS_DIR`, `LOGS_DIR`
- Image: `FACE_IMAGE_SIZE` (160×160)
- Audio: `SAMPLE_RATE`, `N_MFCC`, `N_MELS`, `MAX_AUDIO_LENGTH`
- Model: `FACE_BACKBONE`, `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`
- Age: `AGE_GROUPS`, `AGE_THRESHOLD` (18)
- Liveness: `CAPTCHA_MATCH_THRESHOLD`, `BLINK_THRESHOLD`, `LIP_SYNC_THRESHOLD`
- API: `API_HOST`, `API_PORT`

---

## 📄 License

MIT. See LICENSE for details.

---

**Built with PyTorch, FastAPI, MediaPipe, and Whisper**

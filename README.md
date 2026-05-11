# Multimodal Age Authentication System

A **multimodal deep learning** system for **live age verification** using **face video** and **voice**, with **liveness checks** (numeric voice captcha, blink, lip-sync, face motion). Built with **PyTorch**, **FastAPI**, and a **browser-based** verification UI.

## What the user sees (end-to-end output)

The **web application does not show a numeric age**, an age estimate, or an age group label to the user.

After verification, the UI presents only:

| Role | Top status badge | Age card (main message) |
|------|------------------|-------------------------|
| **Adult** (18+ gate passed) | **VERIFIED** (green) | **Adult** with **Verified** (green styling) |
| **Minor** (under threshold) | **UNVERIFIED** (red) | **Minor** with **Unverified** (red styling) |
| Session failed (captcha / liveness / etc.) | **VERIFICATION FAILED** | Error details as returned by the API |

Liveness and confidence **bars** (face liveness, voice liveness, lip sync, eye blink, overall confidence) are still shown as supporting metrics; they are **not** a displayed age.

Internally, the backend may compute continuous age and fuse face + voice signals to derive **`is_adult`**; that internal value is **not exposed as a number** in the bundled frontend—only **Adult vs Minor** and **verified vs unverified** status, consistent with the project specification (*Multimodal Age Authentication*).

**Note:** This is a **research demonstration** system. The **Adult / Minor** decision reflects model-based inference (apparent cues from face and voice), not legal proof of identity or chronological age. Do not use as a sole gate for regulated access without further legal, security, and fairness review.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.103+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Features

### User-facing goal

- **Binary outcome in the UI**: **Adult** or **Minor**, each paired with **Verified** or **Unverified** status and green/red presentation as described above.
- **No age number** is shown to the user in the default frontend.

### Multimodal inference (backend — drives `is_adult`, not shown as a number in UI)

- **Face**: EfficientNet-based CNN (ImageNet pretrained) → internal continuous age, age-group logits, adult signal.
- **Voice**: MFCC + Mel spectrogram CNN → same style internal outputs.
- **Fusion (optional)**: Multimodal model; disabled by default in favor of the combined unimodal pipeline below.

**Default inference** (`USE_FUSION_FOR_AGE = False` in `src/config.py`): combined internal estimate **0.6 × face age + 0.4 × voice age** when both modalities are available; **`is_adult`** is derived using **`ADULT_THRESHOLD`** (default 18.0) on that fused value (or via the fusion model when enabled). The UI consumes **`is_adult`** and **`success`** only for the Adult/Minor and Verified/Unverified presentation—not the raw age value.

### Liveness

- **Voice**: **Numeric captcha**—user speaks a **5-digit** code; **Whisper** (or fallback STT) transcribes and matches digits (`src/liveness/captcha.py`, `src/liveness/voice_liveness.py`).
- **Face**: Face presence, motion / texture signals (`src/liveness/face_liveness.py`).
- **Eye blink**: MediaPipe-based blink score (`src/liveness/eye_blink.py`).
- **Lip sync**: Audio–video consistency score (`src/liveness/lip_sync.py`).

### Web UI (`frontend/`)

- **Start Verification** → `POST /api/process` with `action=start` → shows **numeric captcha** (digits to speak aloud).
- User records **≥ 5 s** (configurable in `app.js`) with face visible, then **Stop & Verify** → `action=verify` with video + audio.
- **Display policy**: **only** Adult or Minor, with **Verified** / **Unverified** and the status badge rules in the table at the top of this README—**no estimated age, no age in years, no child/teen/adult/senior label** in the results panel.
- After a **successful** session (`success: true` from API):
  - **Adult** (`is_adult`): **VERIFIED** (green) + **Adult** / **Verified**.
  - **Minor** (`is_adult` false): **UNVERIFIED** (red) + **Minor** / **Unverified**.
- **Failed** verification: **VERIFICATION FAILED** and API error message.

The JSON API may still include fields such as `estimated_age` for other clients, logging, or research; the **bundled frontend intentionally does not render numeric age**—only **Adult/Minor** and **verified status** as specified for this project.

---

## Project structure

```
age-authentication/
├── src/
│   ├── api/
│   │   ├── main.py           # FastAPI app, CORS, static mount, lifespan
│   │   ├── routes.py         # /api/status, /api/process
│   │   └── services.py       # Verification pipeline, model orchestration
│   ├── data/                 # Datasets & feature extraction
│   ├── models/               # Face / voice / fusion PyTorch modules
│   ├── liveness/             # Captcha, voice STT, face, blink, lip-sync
│   ├── training/             # Training entrypoints used by run.py
│   └── config.py             # Paths, thresholds, captcha length, API port
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js                # Media capture, /api/process flow, results UI
├── scripts/                  # Optional utilities (e.g. manifests)
├── image-data/UTKFace/       # Optional: face training data (not in repo)
├── audio-data/               # Optional: Common Voice (not in repo)
├── models/                   # Checkpoints (not in repo; place .pth here)
├── run.py                    # CLI: server | train | analyze
├── requirements.txt
└── README.md
```

Optional / experimental areas (if present in your clone): `src/cluster_age/`, `fusion_comparison/`, `face-model-by-teammate/`—not required to run the main server + default UI.

---

## Setup and run

### 1. Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Models

Place checkpoints under `models/` (created automatically):

- `face_age_model_best.pth`
- `voice_age_model_best.pth`  
Optional: `fusion_age_model_best.pth` or `models/fusion/fusion_age_model_best.pth` if `USE_FUSION_FOR_AGE = True`.

Check loaded modules: **GET** `http://localhost:8000/api/status`.

### 3. Training data (optional)

- **Face**: UTKFace under `image-data/UTKFace/` (filename encodes age, e.g. `[age]_[gender]_[race]_[date].jpg`).
- **Audio**: Mozilla Common Voice under `audio-data/` (CSVs + audio; age categories mapped in `src/config.py` → `AUDIO_AGE_MAP`).

```bash
python run.py analyze
```

### 4. Train (optional)

```bash
python run.py train face
python run.py train face --resume

python run.py train voice
python run.py train voice --resume

python run.py train fusion
```

See `run.py --help` for fusion options (epochs, batch size, ablations, etc.).

### 5. Start the API and UI

```bash
python run.py server
```

| URL | Purpose |
|-----|--------|
| http://localhost:8000 | Serves `frontend/index.html` |
| http://localhost:8000/docs | OpenAPI (Swagger) |
| http://localhost:8000/health | Health check |
| http://localhost:8000/static/... | Static assets from `frontend/` |

Default bind: `API_HOST` / `API_PORT` in `src/config.py` (typically `0.0.0.0:8000`).

The frontend uses `API_BASE = 'http://localhost:8000'` in `frontend/app.js`; change it if you host the API elsewhere.

---

## Verification workflow

1. User opens **http://localhost:8000**, grants **camera** and **microphone**.
2. **Start Verification** → `POST /api/process` with form field `action=start` → response includes `captcha_id` and `captcha_sentence` (**digits only**).
3. User speaks the digits clearly (Whisper is configured for English in `voice_liveness.py`; speak digit-by-digit if needed).
4. **Stop & Verify** → `POST /api/process` with `action=verify`, `captcha_id`, `video_file` (WebM), `audio_file` (WAV), `sample_rate`, `duration`.
5. Server validates captcha, runs liveness and multimodal inference, returns JSON with `success`, `is_adult`, `confidence`, `checks` (face/voice liveness, lip_sync, eye_blink, etc.). Numeric age may exist in the payload but **is not shown** in the default UI.
6. UI maps **success + is_adult** → **VERIFIED** + **Adult / Verified**; **success + not adult** → **UNVERIFIED** + **Minor / Unverified**; **failure** → **VERIFICATION FAILED**—with **no age number** displayed, per project design.

---

## Main API (summary)

| Method | Endpoint | Description |
|--------|----------|---------------|
| GET | `/` | Serves main frontend |
| GET | `/health` | Health check |
| GET | `/api/status` | Which models are loaded |
| POST | `/api/process` | `action=start` → captcha; `action=verify` → full verification |

Full schemas: **http://localhost:8000/docs**.

---

## Configuration (`src/config.py`)

| Area | Examples |
|------|----------|
| Paths | `IMAGE_DATA_DIR`, `AUDIO_DATA_DIR`, `MODELS_DIR`, `LOGS_DIR` |
| Image | `FACE_IMAGE_SIZE` (160×160) |
| Audio | `SAMPLE_RATE` (16000), `N_MFCC`, `N_MELS`, `MAX_AUDIO_LENGTH` |
| Age | `AGE_GROUPS`, `AGE_THRESHOLD`, `ADULT_THRESHOLD` |
| Fusion | `USE_FUSION_FOR_AGE` |
| Liveness | `CAPTCHA_MATCH_THRESHOLD`, `CAPTCHA_NUMERIC_LENGTH` (5), `CAPTCHA_NUMERIC_MATCH_THRESHOLD`, `BLINK_THRESHOLD`, `LIP_SYNC_THRESHOLD` |
| API | `API_HOST`, `API_PORT`, `MAX_UPLOAD_SIZE` |

---

## Models (high level)

- **Face**: EfficientNet-B0 backbone → heads for age regression, multi-class age group, adult probability.
- **Voice**: MFCC + Mel tensor → CNN → same style heads.
- **Fusion (optional)**: Single network combining face and voice; enable only if checkpoint and config align.

**Why default is 0.6×face + 0.4×voice (not fusion):** Unimodal models train on large modality-specific datasets; combining them is often more stable and efficient than a fusion model trained on imperfectly paired data. Tune in your own experiments.

---

## Datasets (training)

| Dataset | Location | Role |
|---------|----------|------|
| UTKFace | `image-data/UTKFace/` | Face age supervision |
| Mozilla Common Voice | `audio-data/` | Voice + age buckets → numeric via `AUDIO_AGE_MAP` |
| Kids audio (optional) | `kids-audio-data/output/` | Extra child-like speech |

---

## Limitations (important for reports and demos)

- **Not legal age verification**—models estimate **apparent** age; errors and biases depend on data and demographics.
- **Security**: Open CORS, no API keys, in-memory captcha store—suitable for **local / lab** demos, not hardened production.
- **Dependencies** are heavy (PyTorch, TensorFlow stack where used, Whisper, MediaPipe); document the Python version and GPU/CPU setup you used.

---

## License

MIT (badge above). Add a `LICENSE` file in the repository if your institution requires an explicit copy.

---

Built with **PyTorch**, **FastAPI**, **MediaPipe**, and **Whisper**.

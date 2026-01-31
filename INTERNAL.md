# Internal Technical Reference — Age Authentication System

**Audience:** Internal engineering / ML team.  
**Purpose:** Model specs, datasets, data prep, training flow, APIs, and component design.

---

## 1. Configuration (`src/config.py`)

| Category | Key | Default | Notes |
|----------|-----|---------|--------|
| Paths | `IMAGE_DATA_DIR` | `image-data/UTKFace` | UTKFace images |
| | `AUDIO_DATA_DIR` | `audio-data` | Common Voice root |
| | `KIDS_AUDIO_DATA_DIR` | `kids-audio-data/output` | Kids audio (optional) |
| | `MODELS_DIR` | `models` | Checkpoints |
| | `LOGS_DIR` | `logs` | Training logs |
| Image | `FACE_IMAGE_SIZE` | (160, 160) | Input to face model |
| Audio | `SAMPLE_RATE` | 16000 | Hz |
| | `N_MFCC` | 40 | MFCC coeffs (×3 with deltas → 120) |
| | `N_MELS` | 128 | Mel bands |
| | `HOP_LENGTH` | 512 | Frames |
| | `N_FFT` | 2048 | |
| | `MAX_AUDIO_LENGTH` | 10 | Seconds |
| Age | `AGE_GROUPS` | child/teen/adult/senior | Ranges in docstring |
| | `AGE_THRESHOLD` | 18 | Binary adult |
| Face model | `FACE_BACKBONE` | efficientnet_b0 | efficientnet_b2, resnet50, resnet34 |
| | `FACE_DROPOUT` | 0.3 | |
| Voice model | `VOICE_DROPOUT` | 0.3 | |
| Fusion | `FUSION_HIDDEN_DIM` | 256 | |
| | `FUSION_DROPOUT` | 0.4 | |
| Training | `BATCH_SIZE` | 32 | |
| | `EPOCHS` | 50 | |
| | `LEARNING_RATE` | 1e-4 | |
| | `WEIGHT_DECAY` | 1e-5 | |
| Liveness | `CAPTCHA_MATCH_THRESHOLD` | 0.7 | Captcha overlap |
| | `BLINK_THRESHOLD` | 0.2 | |
| | `LIP_SYNC_THRESHOLD` | 0.6 | |
| API | `API_HOST` | 0.0.0.0 | |
| | `API_PORT` | 8000 | |

---

## 2. Datasets

### 2.1 UTKFace (Face)

- **Path:** `IMAGE_DATA_DIR` = `image-data/UTKFace/`
- **Filename:** `[age]_[gender]_[race]_[date&time].jpg` (e.g. `25_1_2_20170117150557345.jpg`)
- **Parsing:** `DataAnalyzer.analyze_utkface()` — age from first segment; gender 0/1; race 0–4; age_group and is_adult derived.
- **Splits:** Stratified by age_group (e.g. 85% train+val, 15% test; then ~85/15 train/val) via `get_stratified_splits()` in `data_analysis.py`.
- **Output columns:** filepath, filename, age, gender, race, age_group, is_adult.

### 2.2 Mozilla Common Voice (Audio)

- **Path:** `AUDIO_DATA_DIR`; CSVs e.g. `cv-other-train.csv`, `cv-other-dev.csv`, `cv-other-test.csv`.
- **Age:** Categorical (teens, twenties, …) mapped via `AUDIO_AGE_MAP` in config to numeric (e.g. twenties → 24).
- **Parsing:** `DataAnalyzer.analyze_common_voice()` — filter rows with age, add numeric_age, age_group, is_adult; filter by file existence (direct or nested paths).
- **Splits:** Can use dataset split (train/dev/test) or stratified by age_group.
- **Kids audio:** Optional; `KIDS_AUDIO_DATA_DIR`; file format e.g. `F10_01_01.wav`; age range `KIDS_AGE_RANGE` (e.g. 5–12).

### 2.3 Fusion (Age-Matched Pairs)

- **No identity-paired data.** `FusionDataset` pairs by age:
  - Index audio by numeric age; index face by age_group.
  - For each face sample: try exact age → within `age_tolerance` (e.g. 5) → same age_group → random.
- **Labels:** Always from the **face** sample (age, age_group, is_adult).
- **Output:** face image tensor, voice feature tensor (same format as voice training), same labels.

---

## 3. Data Preparation & Transforms

### 3.1 Face

- **Module:** `src/data/face_dataset.py` — `UTKFaceDataset`, `get_face_dataloaders`.
- **Load:** PIL, RGB.
- **Validation/Test:** Resize to `FACE_IMAGE_SIZE` (160×160), ToTensor, Normalize (ImageNet: mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]).
- **Training:** Same normalize + augmentation: resize 1.1× → RandomCrop(160×160), RandomHorizontalFlip(0.5), RandomRotation(15°), ColorJitter, RandomGrayscale(0.1), RandomErasing(0.1).
- **Batch:** `image` (B, 3, 160, 160), `age`, `age_group`, `is_adult`.

### 3.2 Audio (Training)

- **Module:** `src/data/audio_dataset.py` — `AudioFeatureExtractor`, `CommonVoiceDataset`, `get_audio_dataloaders`, `audio_collate_fn`.
- **Load:** librosa 16 kHz mono; trim silence (top_db=20); pad/truncate to `MAX_AUDIO_LENGTH * SAMPLE_RATE` samples.
- **MFCC:** 40 coeffs + delta + delta2 → (120, max_frames). Per-row normalize. Pad/truncate time to `max_frames = int(MAX_AUDIO_LENGTH * SAMPLE_RATE / HOP_LENGTH)`.
- **Mel:** 128 bins, log, normalize; same time axis → (128, max_frames).
- **Combined:** Pad MFCC to 128 rows: `mfcc_padded = zeros((N_MELS, T)); mfcc_padded[:120,:] = mfcc`; then `stack([mfcc_padded, mel], axis=0)` → (2, 128, max_frames). Caching to `cache/audio_features` for non-augmented.
- **Augmentation (train):** time_stretch, pitch_shift, additive noise, gain.
- **Batch:** `features` (B, 2, 128, T), `age`, `age_group`, `is_adult`.

### 3.3 Audio (Inference)

- **Module:** `src/data/audio_dataset.py` — `LiveAudioProcessor.process_audio_bytes()`.
- **Input:** bytes or numpy array; resample to 16 kHz, mono.
- **Same MFCC/Mel pipeline** as training; **same MFCC padding to 128 rows**; stack → (2, 128, T); add batch → (1, 2, 128, T). Must match training exactly so voice model gets (B, 2, 128, T).

### 3.4 Fusion Data Loader

- **Module:** `src/data/fusion_dataset.py` — `FusionDataset`, `fusion_collate_fn`, `get_fusion_dataloaders`.
- **Per sample:** face from `UTKFaceDataset`, voice from `CommonVoiceDataset` (age-matched index). Collate: `face` (B, 3, 160, 160), `voice` (B, 2, 128, T), plus labels and paths.

---

## 4. ML/DL Models

### 4.1 Face Model (`src/models/face_model.py`)

- **Class:** `FaceAgeModel`.
- **Backbone:** TorchVision CNN; default **EfficientNet-B0** (ImageNet pretrained). Classifier head replaced with `nn.Identity()`. Alternatives: EfficientNet-B2, ResNet50, ResNet34.
- **Feature processor:** Linear(1280→512) → BN → ReLU → Dropout → Linear(512→256) → BN → ReLU → Dropout. Output dim 256.
- **Heads:**
  - Age: Linear(256→128) → ReLU → Dropout → Linear(128→1); clamp [0, 100].
  - Age group: Linear(256→128) → ReLU → Dropout → Linear(128→4).
  - Adult: Linear(256→64) → ReLU → Dropout → Linear(64→1); BCEWithLogits.
- **Loss:** `FaceAgeLoss` — age: MAE + 0.5×MSE; group: CrossEntropy; adult: BCEWithLogits; weighted sum.
- **Factory:** `create_face_model()`, `load_face_model(checkpoint_path)`.

### 4.2 Voice Model (`src/models/voice_model.py`)

- **Class:** `VoiceAgeModel`.
- **Input shape:** (B, 2, 128, T). Channel 0: MFCC padded to 128 rows; channel 1: Mel (128, T).
- **Encoder:** 2D CNN (no LSTM):
  - Block 0: Conv2d 2→16, BN, ReLU, MaxPool(2,2), Dropout2d.
  - Block 1: DepthwiseSeparableConv 16→32, SqueezeExcitation(32), pool, dropout.
  - Block 2: DepthwiseSeparableConv 32→64, SE, **FrequencyAttention** (freq dim), pool, dropout.
  - Block 3: DepthwiseSeparableConv 64→128, SE, AdaptiveAvgPool2d(1,1).
- **Feature processor:** Linear(128→96) → BN → ReLU → Dropout → Linear(96→64) → BN → ReLU → Dropout. Output 64.
- **Heads:** Same structure as face: age (regression, clamp 0–100), age_group (4), adult (binary).
- **Loss:** `VoiceAgeLoss` — age: Smooth L1; group: CE; adult: BCE; weighted sum.
- **Factory:** `create_voice_model()`, `load_voice_model(checkpoint_path)`.

### 4.3 Fusion Model (`src/models/fusion_model.py`)

- **Class:** `MultimodalFusionModel`.
- **Encoders:** Same `FaceAgeModel` and `VoiceAgeModel`; loaded from checkpoints; optionally frozen (`freeze_encoders=True`).
- **Projection:** Face 256 → hidden_dim; Voice 64 → hidden_dim (Linear + LayerNorm + ReLU).
- **Cross-modal attention:** `CrossModalAttention` — MultiheadAttention face→voice and voice→face; residual + LayerNorm.
- **Gated fusion:** `GatedFusion` — project face/voice to hidden_dim; gate = sigmoid(MLP(concat(face_h, voice_h))); out = gate*face_h + (1-gate)*voice_h.
- **Fusion processor:** Concat [face_attended, voice_attended, gated] → Linear(3*hidden_dim → 2*hidden_dim) → … → hidden_dim.
- **Heads:** Age (regression), age_group (4), adult (binary); optional face/voice confidence heads.
- **Loss:** `FusionLoss` — MAE + 0.5×MSE (age), CE (group), BCE (adult); optional consistency term (MAE between face age and voice age).
- **Factory:** `create_fusion_model(face_checkpoint, voice_checkpoint, freeze_encoders)`, `load_fusion_model(checkpoint_path)`.

---

## 5. Training Workflow

- **Entry:** `run.py` → `train face | train voice | train fusion` (optional `--resume`).
- **Trainer base:** `src/training/trainer.py` — `Trainer` (train_epoch, validate, save/load checkpoint, CosineAnnealingLR, early stopping by val loss).
- **Face:** `FaceTrainer._forward_batch(batch)` — `batch['image']` → model → criterion(predictions, targets). `train_face_model()` uses DataAnalyzer + get_face_dataloaders, create_face_model, FaceAgeLoss; 12 epochs; early_stopping=5; checkpoint `face_age_model_best.pth`.
- **Voice:** `VoiceTrainer._forward_batch(batch)` — `batch['features']` → model → criterion. `train_voice_model()` uses DataAnalyzer + get_audio_dataloaders, create_voice_model, VoiceAgeLoss; 12 epochs; checkpoint `voice_age_model_best.pth`.
- **Fusion:** `FusionTrainer._forward_batch(batch)` — `batch['face']`, `batch['voice']` → model(face, voice) → criterion. `train_fusion_model()` uses get_fusion_dataloaders, create_fusion_model(face_checkpoint, voice_checkpoint, freeze_encoders=True); 8 epochs; early_stopping=3; checkpoint `fusion_age_model_best.pth`.

---

## 6. Inference / API Workflow

1. **Server:** `run.py server` → `src.api.main.run_server()` (uvicorn on API_HOST:API_PORT). Serves `/` (frontend), `/static`, `/health`, `/api/*`.
2. **Captcha:** Client calls `GET /api/captcha` → service generates sentence and session id → returned to client.
3. **Process:** Client sends `POST /api/process` with video frames and audio (e.g. base64 or files). Routes decode and call `services.process_authentication(frames, audio, sample_rate, …)`.
4. **Service pipeline:**
   - Captcha verification (expected text vs Whisper transcription); voice liveness score.
   - Face: detect/crop with FaceLivenessDetector; optional temporal smoothing (median over frames) in `_predict_age`; single-frame path in `process_authentication` uses last frame → `predict_face_age()` → age, age_group, is_adult.
   - Voice: if transcription present, `LiveAudioProcessor.process_audio_bytes()` → (1, 2, 128, T) → `predict_voice_age()` → age, age_group, is_adult.
   - Fusion in service: when both face_age and voice_age exist, `estimated_age = 0.6*face_age + 0.4*voice_age`; else use whichever is available.
   - Eye blink: EyeBlinkDetector on frames → blink score.
   - Lip sync: LipSyncVerifier on frames + audio → confidence.
   - Success rule: captcha verified and face_age is not None (and any other policy in routes). Response: success, estimated_age, is_adult, confidence, checks (face_detected, face_liveness, blink_detected, captcha_verified, voice_liveness, lip_sync, eye_blink).

---

## 7. API Reference (Internal)

| Method | Path | Handler | Purpose |
|--------|------|---------|--------|
| GET | `/` | main.root | Serve frontend index.html |
| GET | `/health` | main.health_check | Health, version, models_loaded |
| GET | `/api/captcha` | routes (query: complexity) | Generate captcha sentence and session id |
| POST | `/api/process` | routes.process_authentication | Full verification: body with video frames + audio; returns success, estimated_age, confidence, checks |

Request/response schemas: see `src/api/routes.py` and Swagger at `/docs`. Decoding helpers: `_decode_image`, `_decode_base64_image`, `_decode_video`, `_decode_base64_frames`, `_decode_audio`, `_decode_base64_audio`.

---

## 8. Component Map

| Component | Location | Role |
|-----------|----------|------|
| Config | `src/config.py` | Paths, hyperparams, age groups, thresholds |
| Data analysis | `src/data/data_analysis.py` | Parse UTKFace/Common Voice/kids; stratified splits; prepare_face_data, prepare_audio_data |
| Face dataset | `src/data/face_dataset.py` | UTKFaceDataset, get_face_dataloaders, LiveFaceDataset |
| Audio dataset | `src/data/audio_dataset.py` | AudioFeatureExtractor, CommonVoiceDataset, LiveAudioProcessor, get_audio_dataloaders |
| Fusion dataset | `src/data/fusion_dataset.py` | FusionDataset, get_fusion_dataloaders, fusion_collate_fn |
| Face model | `src/models/face_model.py` | FaceAgeModel, FaceAgeLoss, create/load_face_model |
| Voice model | `src/models/voice_model.py` | VoiceAgeModel, VoiceAgeLoss, create/load_voice_model |
| Fusion model | `src/models/fusion_model.py` | MultimodalFusionModel, GatedFusion, CrossModalAttention, FusionLoss, create/load_fusion_model |
| Trainer | `src/training/trainer.py` | Trainer, FaceTrainer, VoiceTrainer, FusionTrainer, train_face_model, train_voice_model, train_fusion_model |
| API app | `src/api/main.py` | FastAPI app, CORS, static mount, lifespan (load services) |
| Routes | `src/api/routes.py` | /api/captcha, /api/process; decode helpers |
| Services | `src/api/services.py` | AgeAuthenticationService: captcha, predict_face_age, predict_voice_age, _predict_age, process_authentication, liveness orchestration |
| Liveness | `src/liveness/*` | face_liveness, voice_liveness (Whisper), lip_sync, eye_blink, captcha |

---

## 9. Checkpoints

- **Face:** `models/face_age_model_best.pth` — `model_state_dict`, optimizer, scheduler, epoch, best_val_loss, history.
- **Voice:** `models/voice_age_model_best.pth` — same structure.
- **Fusion:** `models/fusion_age_model_best.pth` — full fusion model state; encoders can be loaded separately for create_fusion_model.

Loading: `load_face_model(path)`, `load_voice_model(path)`, `load_fusion_model(path)`; services load these in `_load_models()` at startup.

---

## 10. CLI Summary

```bash
python run.py server              # Start API (default 0.0.0.0:8000)
python run.py train face          # Train face model
python run.py train face --resume # Resume face training
python run.py train voice         # Train voice model
python run.py train voice --resume
python run.py train fusion       # Train fusion (uses face + voice checkpoints)
python run.py analyze            # Run full data analysis (UTKFace + Common Voice)
```

---

*Last updated for codebase as of INTERNAL.md creation. For exact shapes and layer sizes, refer to the source files listed above.*

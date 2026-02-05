# Fusion vs existing age comparison

This folder contains **new** code only. No existing project files are modified.

## Purpose

- Run the **existing** age pipeline (0.6×face + 0.4×voice, or single modality).
- Run the **fusion model** on the same video + audio.
- Compare both ages (via **frontend** or **CLI script**).

---

## Test through frontend (recommended)

1. **Start the server with fusion comparison enabled** (from project root):

   ```bash
   python fusion_comparison/server.py
   ```

   This runs the same app as `python run.py server`, plus the fusion endpoint and the comparison page.

2. **Open in the browser:**

   - **Normal app:** http://localhost:8000  
   - **Compare page:** http://localhost:8000/compare  

3. **On the compare page:**

   - Click **Start Verification** → you get a captcha (same as main app).
   - Read the sentence aloud while the camera records (at least 5 seconds).
   - Click **Stop & Compare** → the request goes to `/api/process-with-fusion`.
   - You see **Existing (weighted)** age, **Fusion model** age, and the **difference**.

Same flow as the main app; only the verify step uses the fusion endpoint and shows both ages.

---

## CLI script (video + audio files)

From the **project root**:

```bash
python fusion_comparison/compare_age.py --video path/to/video.mp4 --audio path/to/audio.wav
```

- **Video:** Any format OpenCV can read (e.g. .mp4). One frame is used (the last frame).
- **Audio:** WAV or other format. Should be the same recording as the video.

The script prints: **Existing** age, **Fusion** age, and the difference.

---

## Requirements

- Same as main project (PyTorch, OpenCV, etc.).
- Checkpoints in `models/`:
  - `face_age_model_best.pth` and `voice_age_model_best.pth` for the existing pipeline.
  - `fusion_age_model_best.pth` for fusion. If missing, the compare page/script still runs and reports existing only; fusion_age will be null.

---

## Summary

| Goal              | Command / URL                                      |
|-------------------|----------------------------------------------------|
| Run app + compare | `python fusion_comparison/server.py`               |
| Compare page      | http://localhost:8000/compare                       |
| Normal app        | http://localhost:8000 (when using server above)    |
| CLI comparison    | `python fusion_comparison/compare_age.py --video … --audio …` |

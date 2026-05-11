"""
Phase 2 — Preprocessing: load, resample, mono, trim silence, normalize.
Voice-only; no language or demographic input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

from src.config import SAMPLE_RATE
from src.cluster_age.config import SEGMENT_LENGTH_SEC

# Max length for single file (seconds); pad/truncate to this
MAX_AUDIO_LENGTH_SEC = 10.0


def load_audio(
    source: Union[str, Path, np.ndarray, bytes],
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Load audio from path or bytes. If already ndarray, assume it is at sample_rate.
    Returns mono float32 waveform, shape (n_samples,).
    """
    import librosa
    if isinstance(source, np.ndarray):
        if source.dtype != np.float32:
            source = source.astype(np.float32)
        if source.ndim > 1:
            source = source.mean(axis=1)
        return source
    if isinstance(source, (str, Path)):
        y, sr = librosa.load(str(source), sr=sample_rate, mono=True)
        return y.astype(np.float32)
    if isinstance(source, bytes):
        import io
        y, sr = librosa.load(io.BytesIO(source), sr=sample_rate, mono=True)
        return y.astype(np.float32)
    raise TypeError(f"Unsupported source type: {type(source)}")


def trim_silence(y: np.ndarray, top_db: int = 20) -> np.ndarray:
    """Trim leading/trailing silence. Voice-only preprocessing."""
    import librosa
    y, _ = librosa.effects.trim(y, top_db=top_db)
    return y


def resample_if_needed(y: np.ndarray, orig_sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Resample to target_sr if different."""
    if orig_sr == target_sr:
        return y
    import librosa
    return librosa.resample(y.astype(np.float64), orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)


def normalize_peak(y: np.ndarray) -> np.ndarray:
    """Peak-normalize to [-1, 1]. In-place friendly."""
    peak = np.abs(y).max()
    if peak > 1e-8:
        y = y / peak
    return y


def pad_or_truncate(y: np.ndarray, max_samples: int) -> np.ndarray:
    """Truncate or zero-pad to max_samples."""
    if len(y) > max_samples:
        return y[:max_samples].copy()
    if len(y) < max_samples:
        return np.pad(y, (0, max_samples - len(y)), mode="constant", constant_values=0.0)
    return y


def preprocess(
    source: Union[str, Path, np.ndarray, bytes],
    sample_rate: int = SAMPLE_RATE,
    trim: bool = True,
    max_length_sec: float = MAX_AUDIO_LENGTH_SEC,
) -> np.ndarray:
    """
    Full preprocessing: load -> trim silence -> peak-normalize -> pad/truncate to max_length_sec.
    Returns mono float32 waveform at sample_rate.
    """
    y = load_audio(source, sample_rate)
    if trim and len(y) > 0:
        y = trim_silence(y)
    y = normalize_peak(y)
    max_samples = int(max_length_sec * sample_rate)
    y = pad_or_truncate(y, max_samples)
    return y

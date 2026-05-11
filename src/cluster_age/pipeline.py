"""
Phase 2 — Training vs inference pipeline separation.
Scaler fit only on A; spoof gate before age path at inference.
Embedding stage receives consistent scaled feature vector (FEATURE_DIM,) per utterance.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.config import SAMPLE_RATE
from src.cluster_age.config import (
    SEGMENT_LENGTH_SEC,
    SEGMENT_MIN_SEC,
    SCALER_ARTIFACT_NAME,
)
from src.cluster_age.preprocessing import preprocess
from src.cluster_age.segmentation import segment_waveform, segment_waveform_single_or_empty
from src.cluster_age.features import extract_utterance_features, FEATURE_DIM
from src.cluster_age.spoof_gate import SpoofGateBase, SpoofGateStub
from src.cluster_age.scaler_protocol import fit_scaler_on_vectors, transform_vectors, save_scaler, load_scaler


# ---------------------------------------------------------------------------
# Training pipeline (batch from manifest; scaler fit on A only)
# ---------------------------------------------------------------------------


def load_audio_for_training(path: Union[str, Path]) -> Optional[np.ndarray]:
    """Load and preprocess one file. Returns None on failure."""
    try:
        return preprocess(path, sample_rate=SAMPLE_RATE)
    except Exception:
        return None


def training_extract_utterance(
    audio_path: Union[str, Path],
    segment_length_sec: float = SEGMENT_LENGTH_SEC,
    min_length_sec: float = SEGMENT_MIN_SEC,
) -> Optional[np.ndarray]:
    """
    For one training sample: load -> preprocess -> segment -> extract -> one vector.
    Returns shape (FEATURE_DIM,) or None if invalid/short.
    """
    y = load_audio_for_training(audio_path)
    if y is None or len(y) < int(min_length_sec * SAMPLE_RATE):
        return None
    segments = segment_waveform(
        y,
        sample_rate=SAMPLE_RATE,
        segment_length_sec=segment_length_sec,
        min_length_sec=min_length_sec,
    )
    if not segments:
        return None
    return extract_utterance_features(segments, sr=SAMPLE_RATE)


def training_batch_from_manifest(
    manifest_path: Union[str, Path],
    max_samples: Optional[int] = None,
) -> np.ndarray:
    """
    Load manifest (CSV with 'path' column), extract features for each row, return (N, FEATURE_DIM).
    Used for Dataset A to fit scaler; optionally cap N for memory.
    """
    df = pd.read_csv(manifest_path)
    if "path" not in df.columns:
        raise ValueError("Manifest must have 'path' column")
    paths = df["path"].astype(str).tolist()
    if max_samples is not None:
        paths = paths[:max_samples]
    vectors: List[np.ndarray] = []
    for p in paths:
        if not Path(p).exists():
            continue
        v = training_extract_utterance(p)
        if v is not None:
            vectors.append(v)
    if not vectors:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)
    return np.stack(vectors)


def fit_and_save_scaler_on_a(
    train_manifest_path: Union[str, Path],
    output_dir: Union[str, Path],
    max_samples: Optional[int] = None,
) -> dict:
    """
    Fit scaler on Dataset A only (train manifest). Save to output_dir/SCALER_ARTIFACT_NAME.
    Returns the fitted scaler dict.
    """
    vectors = training_batch_from_manifest(train_manifest_path, max_samples=max_samples)
    if vectors.shape[0] == 0:
        raise ValueError("No valid feature vectors from train manifest; cannot fit scaler")
    scaler = fit_scaler_on_vectors(vectors)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_scaler(scaler, output_dir / SCALER_ARTIFACT_NAME)
    return scaler


# ---------------------------------------------------------------------------
# Inference pipeline (single request; spoof gate -> features -> scaled vector)
# ---------------------------------------------------------------------------


def inference_preprocess_and_segment(
    source: Union[str, Path, np.ndarray, bytes],
    sample_rate: int = SAMPLE_RATE,
    segment_length_sec: float = SEGMENT_LENGTH_SEC,
    min_length_sec: float = SEGMENT_MIN_SEC,
) -> Tuple[Optional[np.ndarray], List[np.ndarray]]:
    """
    Preprocess and segment. Returns (full_waveform_for_spoof, list_of_segments).
    If too short, segments may be empty.
    """
    y = preprocess(source, sample_rate=sample_rate)
    if len(y) < int(min_length_sec * sample_rate):
        return y, []
    segments = segment_waveform(
        y,
        sample_rate=sample_rate,
        segment_length_sec=segment_length_sec,
        min_length_sec=min_length_sec,
    )
    if not segments:
        # Fallback: one segment if above min length
        segments = segment_waveform_single_or_empty(
            y, sample_rate=sample_rate,
            segment_length_sec=segment_length_sec,
            min_length_sec=min_length_sec,
        )
    return y, segments


def inference_run_spoof_gate(
    waveform: np.ndarray,
    sample_rate: int,
    gate: Optional[SpoofGateBase] = None,
) -> Tuple[bool, float]:
    """Run spoof gate. If gate is None, use stub (always pass)."""
    if gate is None:
        gate = SpoofGateStub()
    return gate.run(waveform, sample_rate)


def inference_feature_vector(
    segments: List[np.ndarray],
    scaler: Optional[dict] = None,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Extract and aggregate features; optionally scale. Returns (FEATURE_DIM,) for embedding stage.
    """
    vec = extract_utterance_features(segments, sr=sample_rate)
    if scaler is not None:
        vec = transform_vectors(vec, scaler)
    return vec


def inference_pipeline(
    source: Union[str, Path, np.ndarray, bytes],
    scaler_path: Optional[Union[str, Path]] = None,
    scaler: Optional[dict] = None,
    spoof_gate: Optional[SpoofGateBase] = None,
    sample_rate: int = SAMPLE_RATE,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Full inference path: preprocess -> segment -> spoof gate -> features -> scale -> vector.
    Returns (feature_vector, None) on success. Returns (None, "spoof") if gate fails.
    Returns (None, "too_short") if audio too short for one segment.
    """
    y, segments = inference_preprocess_and_segment(source, sample_rate=sample_rate)
    if not segments:
        return None, "too_short"
    passed, _ = inference_run_spoof_gate(y, sample_rate, gate=spoof_gate)
    if not passed:
        return None, "spoof"
    if scaler is None and scaler_path is not None:
        scaler = load_scaler(scaler_path)
    vec = inference_feature_vector(segments, scaler=scaler, sample_rate=sample_rate)
    return vec, None


# ---------------------------------------------------------------------------
# Embedding-stage interface (consistent input)
# ---------------------------------------------------------------------------


def get_embedding_input_dim() -> int:
    """Fixed dimension that embedding stage receives per utterance (after scaling)."""
    return FEATURE_DIM


def prepare_embedding_input_batch(vectors: np.ndarray, scaler: Optional[dict] = None) -> np.ndarray:
    """
    For training: (N, FEATURE_DIM) -> scale if scaler provided -> (N, FEATURE_DIM).
    Embedding model receives this batch.
    """
    if scaler is not None:
        vectors = transform_vectors(vectors, scaler)
    return vectors.astype(np.float32)


def prepare_embedding_input_single(vector: np.ndarray, scaler: Optional[dict] = None) -> np.ndarray:
    """
    For inference: (FEATURE_DIM,) -> scale if scaler provided -> (1, FEATURE_DIM) for model.
    """
    if vector.ndim == 1:
        vector = vector[np.newaxis, :]
    if scaler is not None:
        vector = transform_vectors(vector, scaler)
    return vector.astype(np.float32)

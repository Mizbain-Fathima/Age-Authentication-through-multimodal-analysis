"""
Phase 2 — Segmentation: slice waveform into fixed-length segments.
Min length enforced; segments below are dropped.
"""

from __future__ import annotations

from typing import List

import numpy as np

from src.config import SAMPLE_RATE
from src.cluster_age.config import SEGMENT_LENGTH_SEC, SEGMENT_MIN_SEC, SEGMENT_OVERLAP_SEC


def segment_waveform(
    y: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    segment_length_sec: float = SEGMENT_LENGTH_SEC,
    min_length_sec: float = SEGMENT_MIN_SEC,
    overlap_sec: float = SEGMENT_OVERLAP_SEC,
) -> List[np.ndarray]:
    """
    Split waveform into segments of length segment_length_sec (and optional overlap).
    Drops segments shorter than min_length_sec.
    Returns list of float32 segments, each shape (n_samples,).
    """
    seg_samples = int(segment_length_sec * sample_rate)
    min_samples = int(min_length_sec * sample_rate)
    step = max(1, seg_samples - int(overlap_sec * sample_rate))
    n = len(y)
    segments: List[np.ndarray] = []
    start = 0
    while start + seg_samples <= n:
        seg = y[start : start + seg_samples].astype(np.float32)
        segments.append(seg)
        start += step
    # Last segment if remainder >= min_length_sec
    if start < n and (n - start) >= min_samples:
        seg = y[start:].astype(np.float32)
        if len(seg) < seg_samples:
            seg = np.pad(seg, (0, seg_samples - len(seg)), mode="constant", constant_values=0.0)
        segments.append(seg)
    return segments


def segment_waveform_single_or_empty(
    y: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    segment_length_sec: float = SEGMENT_LENGTH_SEC,
    min_length_sec: float = SEGMENT_MIN_SEC,
) -> List[np.ndarray]:
    """
    For short utterances: if total length >= min_length_sec, return one segment (padded/truncated).
    Otherwise return empty list (caller may treat as invalid).
    """
    min_samples = int(min_length_sec * sample_rate)
    seg_samples = int(segment_length_sec * sample_rate)
    if len(y) < min_samples:
        return []
    if len(y) >= seg_samples:
        return [y[:seg_samples].astype(np.float32)]
    seg = np.pad(y.astype(np.float32), (0, seg_samples - len(y)), mode="constant", constant_values=0.0)
    return [seg]

"""
Phase 2 — Voice-only feature extraction for cluster-age.
MFCC, F0, formants, spectral centroid, energy, HNR, prosody, temporal.
Speech-rate features are down-weighted in aggregation (see config).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from src.config import SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS
from src.cluster_age.config import SPEECH_RATE_WEIGHT, PITCH_FORMANT_WEIGHT


# Fixed feature dimension for embedding interface (after aggregation)
# Layout: [mfcc_stats, f0_stats, formant_stats, spectral_stats, energy_stats, hnr_stats, prosody_stats, temporal_stats]
FEATURE_DIM = 128  # total float features per utterance after aggregation


def _extract_mfcc_stats(y: np.ndarray, sr: int) -> np.ndarray:
    """MFCC mean/std per coefficient, then global mean of those. Returns fixed-size vector."""
    import librosa
    n_mfcc = 13
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=N_FFT, hop_length=HOP_LENGTH)
    if mfcc.size == 0:
        return np.zeros(2 * n_mfcc, dtype=np.float32)
    mean_per_coef = mfcc.mean(axis=1)
    std_per_coef = mfcc.std(axis=1)
    return np.concatenate([mean_per_coef, np.nan_to_num(std_per_coef, nan=0.0)]).astype(np.float32)


def _extract_f0_stats(y: np.ndarray, sr: int) -> np.ndarray:
    """F0 (pitch) stats: mean, std, median. Voice-only."""
    import librosa
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr, frame_length=2048)
    f0 = np.nan_to_num(f0, nan=0.0)
    if f0.size == 0:
        return np.zeros(4, dtype=np.float32)
    return np.array([f0.mean(), f0.std(), np.median(f0), (f0 > 0).mean()], dtype=np.float32)


def _extract_formant_stats(y: np.ndarray, sr: int) -> np.ndarray:
    """Simple formant proxy from LPC spectrum peaks. Returns fixed-size vector."""
    import librosa
    # Use spectral envelope as proxy if full formant tracking not available
    n_fft = min(N_FFT, len(y) // 4)
    if n_fft < 64:
        return np.zeros(6, dtype=np.float32)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=HOP_LENGTH)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    # First few spectral peaks as formant proxy (simplified)
    flat = S.mean(axis=1)
    if flat.size < 10:
        return np.zeros(6, dtype=np.float32)
    out = np.array([
        flat.mean(), flat.std(),
        float(freqs[np.argmax(flat)]) if flat.max() > 0 else 0.0,
        np.percentile(flat, 25), np.percentile(flat, 75), (flat > flat.mean()).mean()
    ], dtype=np.float32)
    return out


def _extract_spectral_centroid_stats(y: np.ndarray, sr: int) -> np.ndarray:
    """Spectral centroid and spread. Voice-only."""
    import librosa
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    spread = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    if cent.size == 0:
        return np.zeros(4, dtype=np.float32)
    return np.array([cent.mean(), cent.std(), spread.mean(), spread.std()], dtype=np.float32)


def _extract_energy_stats(y: np.ndarray) -> np.ndarray:
    """RMS and energy distribution stats."""
    import librosa
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)
    if rms.size == 0:
        return np.zeros(4, dtype=np.float32)
    rms = rms.flatten()
    return np.array([rms.mean(), rms.std(), np.percentile(rms, 25), np.percentile(rms, 75)], dtype=np.float32)


def _extract_hnr(y: np.ndarray, sr: int) -> np.ndarray:
    """Harmonic-to-noise ratio proxy from autocorrelation."""
    import librosa
    # Simplified: use spectral flatness as noise proxy
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    if flatness.size == 0:
        return np.zeros(2, dtype=np.float32)
    flatness = np.nan_to_num(flatness, nan=0.0)
    return np.array([flatness.mean(), flatness.std()], dtype=np.float32)


def _extract_prosody_temporal(y: np.ndarray, sr: int) -> np.ndarray:
    """Prosodic/temporal: duration, zero-crossing rate (tempo proxy), articulation proxy.
    These are the ones we down-weight vs pitch/formants."""
    import librosa
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP_LENGTH)
    duration_sec = len(y) / sr
    if zcr.size == 0:
        return np.array([duration_sec, 0.0, 0.0], dtype=np.float32)
    return np.array([duration_sec, zcr.mean(), zcr.std()], dtype=np.float32)


def extract_segment_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract voice-only features for one segment. Returns 1D float32 vector.
    No language, accent, or demographic input.
    """
    mfcc = _extract_mfcc_stats(y, sr)
    f0 = _extract_f0_stats(y, sr)
    formant = _extract_formant_stats(y, sr)
    spectral = _extract_spectral_centroid_stats(y, sr)
    energy = _extract_energy_stats(y)
    hnr = _extract_hnr(y, sr)
    prosody = _extract_prosody_temporal(y, sr)
    vec = np.concatenate([mfcc, f0, formant, spectral, energy, hnr, prosody]).astype(np.float32)
    return vec


def get_feature_dim() -> int:
    """Return the dimension of a single-segment feature vector (before aggregation)."""
    y_dummy = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)  # 2 s
    return extract_segment_features(y_dummy, SAMPLE_RATE).size


def aggregate_segment_features(
    segment_vectors: List[np.ndarray],
    speech_rate_weight: float = SPEECH_RATE_WEIGHT,
    pitch_formant_weight: float = PITCH_FORMANT_WEIGHT,
) -> np.ndarray:
    """
    Aggregate per-segment features to one vector per utterance.
    Down-weights prosody/temporal (last 3 dims) so speech rate does not dominate.
    Output length is always FEATURE_DIM for embedding interface.
    """
    if not segment_vectors:
        return np.zeros(FEATURE_DIM, dtype=np.float32)
    stack = np.stack(segment_vectors, axis=0)
    n_prosody = 3  # last 3 from _extract_prosody_temporal
    n_main = stack.shape[1] - n_prosody
    main_part = stack[:, :n_main].mean(axis=0)
    prosody_part = stack[:, -n_prosody:].mean(axis=0)
    # Weighted combination so pitch/formants dominate
    combined = np.concatenate([
        main_part * pitch_formant_weight,
        prosody_part * speech_rate_weight,
    ]).astype(np.float32)
    # Pad or truncate to FEATURE_DIM for fixed embedding input
    if combined.size < FEATURE_DIM:
        combined = np.pad(combined, (0, FEATURE_DIM - combined.size), mode="constant", constant_values=0.0)
    elif combined.size > FEATURE_DIM:
        combined = combined[:FEATURE_DIM]
    return combined


def extract_utterance_features(
    segments: List[np.ndarray],
    sr: int = SAMPLE_RATE,
    speech_rate_weight: float = SPEECH_RATE_WEIGHT,
    pitch_formant_weight: float = PITCH_FORMANT_WEIGHT,
) -> np.ndarray:
    """
    Extract features for all segments and aggregate to one vector per utterance.
    Returns shape (FEATURE_DIM,) float32.
    """
    vecs = [extract_segment_features(seg, sr) for seg in segments]
    return aggregate_segment_features(vecs, speech_rate_weight, pitch_formant_weight)

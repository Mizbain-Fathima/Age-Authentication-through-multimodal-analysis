"""
Phase 2 — Feature pipeline config for cluster-age verification.
Uses main SAMPLE_RATE etc. where applicable; defines segment/feature params only.
"""

from src.config import SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS

# Segment settings (Phase 2)
SEGMENT_LENGTH_SEC = 4.0
SEGMENT_MIN_SEC = 1.5
SEGMENT_OVERLAP_SEC = 0.0  # no overlap by default

# Feature aggregation: down-weight speech-rate so pitch/formants dominate (edge-case stability)
# Weights for aggregation: prosody/speech-rate group vs rest. Sum per group normalized to 1.
SPEECH_RATE_WEIGHT = 0.25   # articulation rate, tempo
PITCH_FORMANT_WEIGHT = 0.75  # F0, formants, spectral, energy, HNR

# Scaler: fit only on A (train); persisted artifact name
SCALER_ARTIFACT_NAME = "cluster_age_scaler.npz"

# Latency budget (ms) per stage for p95 ≤ 300 ms total
BUDGET_PREPROCESS_MS = 40
BUDGET_FEATURES_MS = 60
BUDGET_SPOOF_MS = 80

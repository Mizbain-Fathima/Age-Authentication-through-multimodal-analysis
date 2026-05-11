"""
Phase 2 — Scaler protocol: fit only on A (train); persist; load at inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np


def fit_scaler_on_vectors(vectors: np.ndarray) -> dict:
    """
    Fit scaler on training (Dataset A) feature vectors.
    vectors: shape (N, FEATURE_DIM). Returns a dict with 'mean' and 'scale' (std, with 1e-8 floor).
    """
    mean = np.mean(vectors, axis=0)
    scale = np.std(vectors, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return {"mean": mean.astype(np.float32), "scale": scale.astype(np.float32)}


def transform_vectors(vectors: np.ndarray, scaler: dict) -> np.ndarray:
    """Apply (x - mean) / scale. vectors: (N, D) or (D,)."""
    out = (vectors - scaler["mean"]) / scaler["scale"]
    return out.astype(np.float32)


def save_scaler(scaler: dict, path: Union[str, Path]) -> None:
    """Persist scaler to disk (NumPy npz or joblib)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path.with_suffix(".npz"), mean=scaler["mean"], scale=scaler["scale"])


def load_scaler(path: Union[str, Path]) -> dict:
    """Load scaler from disk."""
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    data = np.load(path)
    return {"mean": data["mean"], "scale": data["scale"]}

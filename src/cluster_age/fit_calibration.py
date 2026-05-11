"""
Phase 3 — Fit temperature scaling on Dataset B only.
Uses existing scaler, embedding, and binary head. Optimizes T to minimize NLL on B.
No model weight changes. No use of A or C.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from loguru import logger

from src.cluster_age.config import SCALER_ARTIFACT_NAME
from src.cluster_age.models import EmbeddingMLP, BinaryHead
from src.cluster_age.pipeline import training_extract_utterance, load_scaler, transform_vectors
from src.cluster_age.split_validation import age_to_binary


def load_b_features_and_labels(calibration_manifest: Path, scaler_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load calibration manifest (B), extract and scale features, return (X, y_binary)."""
    import pandas as pd
    df = pd.read_csv(calibration_manifest)
    if "path" not in df.columns or "age" not in df.columns:
        raise ValueError("Calibration manifest must have 'path' and 'age' columns")
    scaler = load_scaler(scaler_path)
    vectors_list = []
    labels_list = []
    for _, row in df.iterrows():
        p = Path(row["path"])
        if not p.exists():
            continue
        v = training_extract_utterance(str(p))
        if v is not None:
            vectors_list.append(v)
            labels_list.append(age_to_binary(int(row["age"])))
    if not vectors_list:
        raise ValueError("No valid feature vectors from calibration manifest")
    X = transform_vectors(np.stack(vectors_list), scaler)
    y = np.array(labels_list, dtype=np.float32)
    return X, y


def fit_temperature(logits: np.ndarray, y: np.ndarray, device: torch.device) -> float:
    """Optimize temperature T to minimize NLL: P(minor) = sigmoid(logit / T). T > 0 via log-param."""
    logits_t = torch.from_numpy(logits).float().to(device)
    y_t = torch.from_numpy(y).float().to(device)
    log_T = torch.nn.Parameter(torch.tensor(0.0, device=device))  # T = exp(log_T) > 0
    optimizer = torch.optim.LBFGS([log_T], lr=0.1, max_iter=100)
    nll_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")

    def closure():
        optimizer.zero_grad()
        T = torch.exp(log_T).clamp(min=1e-4, max=1e4)
        scaled = logits_t / T
        loss = nll_fn(scaled, y_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    T = torch.exp(log_T.detach()).clamp(min=1e-4, max=1e4)
    return float(T.item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: Fit temperature calibration on B.")
    parser.add_argument("--calibration-manifest", type=Path, required=True, help="Path to calibration_manifest.csv (B)")
    parser.add_argument("--artifact-dir", type=Path, required=True, help="Directory with scaler, embedding, head")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path (default: artifact_dir/calibration_params.json)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler_path = args.artifact_dir / SCALER_ARTIFACT_NAME
    if not scaler_path.exists():
        logger.error(f"Scaler not found: {scaler_path}")
        raise SystemExit(1)
    embed_path = args.artifact_dir / "embedding_weights.pt"
    head_path = args.artifact_dir / "binary_head_weights.pt"
    if not embed_path.exists() or not head_path.exists():
        logger.error("embedding_weights.pt and binary_head_weights.pt must exist in artifact-dir")
        raise SystemExit(1)

    X, y = load_b_features_and_labels(args.calibration_manifest, scaler_path)
    logger.info(f"B: {X.shape[0]} samples for calibration")

    embedding = EmbeddingMLP().to(device)
    head = BinaryHead().to(device)
    embedding.load_state_dict(torch.load(embed_path, map_location=device))
    head.load_state_dict(torch.load(head_path, map_location=device))
    embedding.eval()
    head.eval()

    with torch.no_grad():
        x_t = torch.from_numpy(X).float().to(device)
        emb = embedding(x_t)
        logits = head(emb)
        logits_np = logits.cpu().numpy()

    T = fit_temperature(logits_np, y, device)
    logger.info(f"Fitted temperature T = {T:.4f}")

    out_path = args.output or args.artifact_dir / "calibration_params.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    params = {"method": "temperature", "T": T}
    with open(out_path, "w") as f:
        json.dump(params, f, indent=2)
    logger.info(f"Saved {out_path}")


if __name__ == "__main__":
    main()

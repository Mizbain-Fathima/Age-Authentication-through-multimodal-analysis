"""
Phase 3 — Select threshold θ on Dataset C only.
Grid search θ; choose largest θ such that FMR ≤ target_fmr.
Save threshold_theta.json. Fail if no θ satisfies FMR constraint.
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

THETA_MIN = 0.01
THETA_MAX = 0.99
THETA_STEP = 0.01
DEFAULT_TARGET_FMR = 0.01


def load_c_features_and_labels(threshold_manifest: Path, scaler_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load threshold manifest (C), extract and scale features, return (X, y_binary)."""
    import pandas as pd
    df = pd.read_csv(threshold_manifest)
    if "path" not in df.columns or "age" not in df.columns:
        raise ValueError("Threshold manifest must have 'path' and 'age' columns")
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
        raise ValueError("No valid feature vectors from threshold manifest")
    X = transform_vectors(np.stack(vectors_list), scaler)
    y = np.array(labels_list, dtype=np.float32)
    return X, y


def apply_calibration(logits: np.ndarray, T: float) -> np.ndarray:
    """P(minor) = sigmoid(logit / T)."""
    return 1.0 / (1.0 + np.exp(-logits / max(T, 1e-6)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: Select threshold θ on C.")
    parser.add_argument("--threshold-manifest", type=Path, required=True, help="Path to threshold_manifest.csv (C)")
    parser.add_argument("--artifact-dir", type=Path, required=True, help="Directory with scaler, embedding, head, calibration_params.json")
    parser.add_argument("--target-fmr", type=float, default=DEFAULT_TARGET_FMR, help="Max allowed FMR (default 0.01)")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path (default: artifact_dir/threshold_theta.json)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler_path = args.artifact_dir / SCALER_ARTIFACT_NAME
    cal_path = args.artifact_dir / "calibration_params.json"
    embed_path = args.artifact_dir / "embedding_weights.pt"
    head_path = args.artifact_dir / "binary_head_weights.pt"
    for p, name in [
        (scaler_path, "scaler"),
        (cal_path, "calibration_params.json"),
        (embed_path, "embedding_weights.pt"),
        (head_path, "binary_head_weights.pt"),
    ]:
        if not p.exists():
            logger.error(f"Missing {name}: {p}")
            raise SystemExit(1)

    with open(cal_path) as f:
        cal = json.load(f)
    T = cal.get("T", 1.0)

    X, y = load_c_features_and_labels(args.threshold_manifest, scaler_path)
    n_adult = int((1 - y).sum())
    n_minor = int(y.sum())
    logger.info(f"C: {X.shape[0]} samples (minor={n_minor}, adult={n_adult})")

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

    p_minor = apply_calibration(logits_np, T)

    # Grid search: largest θ such that FMR ≤ target_fmr
    # FMR = adult predicted minor / total adult
    # FAR = minor predicted adult / total minor
    thetas = np.arange(THETA_MIN, THETA_MAX + THETA_STEP / 2, THETA_STEP)
    best_theta = None
    best_fmr = None
    best_far = None
    for theta in thetas[::-1]:
        pred_minor = (p_minor >= theta).astype(np.float32)
        # True label: 1 = minor, 0 = adult
        fmr = (1 - y) * pred_minor  # adult predicted minor
        fmr = fmr.sum() / max(n_adult, 1)
        far = y * (1 - pred_minor)  # minor predicted adult
        far = far.sum() / max(n_minor, 1)
        if fmr <= args.target_fmr:
            best_theta = float(theta)
            best_fmr = float(fmr)
            best_far = float(far)
            break

    if best_theta is None:
        logger.error(f"No θ in [{THETA_MIN}, {THETA_MAX}] satisfies FMR ≤ {args.target_fmr}. Failing.")
        raise SystemExit(1)

    logger.info(f"Selected θ = {best_theta:.2f} (FMR on C = {best_fmr:.4f}, FAR on C = {best_far:.4f})")

    out_path = args.output or args.artifact_dir / "threshold_theta.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "threshold_theta": best_theta,
        "fmr_target": args.target_fmr,
        "fmr_on_c": best_fmr,
        "far_on_c": best_far,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Saved {out_path}")


if __name__ == "__main__":
    main()

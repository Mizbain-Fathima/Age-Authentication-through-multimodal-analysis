"""
Phase 3 — Train embedding MLP and binary head on Dataset A only.
Then compute label-based C1–C5 centroids from A embeddings and save.
No use of B/C/D. Binary head is sole classification authority.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger

from src.cluster_age.config import SCALER_ARTIFACT_NAME
from src.cluster_age.models import EmbeddingMLP, BinaryHead, EMBEDDING_DIM
from src.cluster_age.pipeline import (
    training_extract_utterance,
    fit_and_save_scaler_on_a,
    load_scaler,
    transform_vectors,
)
from src.cluster_age.scaler_protocol import fit_scaler_on_vectors, save_scaler
from src.cluster_age.split_validation import age_to_binary, age_to_cluster, add_cluster_column

SEED = 42
ADULT_CLASS_WEIGHT = 2.0
EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_age_18_maps_to_c5(df: pd.DataFrame) -> None:
    """Fail if any row with age >= 18 is not mapped to C5."""
    df = add_cluster_column(df.copy())
    adults = df[df["age"] >= 18]
    if len(adults) == 0:
        return
    not_c5 = adults[adults["cluster"] != "C5"]
    if len(not_c5) > 0:
        logger.error("Age >= 18 must map to C5. Found rows with age>=18 and cluster != C5.")
        sys.exit(1)


def load_a_features_and_labels(
    train_manifest: Path,
    scaler_path: Path | None,
    artifact_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Load train manifest (A), extract 128-d features per row (aligned with age).
    If scaler_path is None, fit scaler on A and save to artifact_dir.
    Returns (scaled_features, binary_labels, cluster_indices, scaler).
    """
    df = pd.read_csv(train_manifest)
    if "path" not in df.columns or "age" not in df.columns:
        logger.error("Train manifest must have 'path' and 'age' columns.")
        sys.exit(1)
    validate_age_18_maps_to_c5(df)

    vectors_list = []
    ages_list = []
    for _, row in df.iterrows():
        p = Path(row["path"])
        if not p.exists():
            continue
        v = training_extract_utterance(str(p))
        if v is not None:
            vectors_list.append(v)
            ages_list.append(int(row["age"]))

    if not vectors_list:
        logger.error("No valid feature vectors from train manifest.")
        sys.exit(1)
    vectors = np.stack(vectors_list)

    if scaler_path is None:
        logger.info("Fitting scaler on A and saving to artifact_dir")
        scaler = fit_scaler_on_vectors(vectors)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        save_scaler(scaler, artifact_dir / SCALER_ARTIFACT_NAME)
    else:
        scaler = load_scaler(scaler_path)

    binary_labels = np.array([age_to_binary(a) for a in ages_list], dtype=np.float32)
    cluster_order = ["C1", "C2", "C3", "C4", "C5"]
    cluster_indices = np.array(
        [cluster_order.index(age_to_cluster(a)) for a in ages_list],
        dtype=np.int64,
    )
    scaled = transform_vectors(vectors, scaler)
    return scaled, binary_labels, cluster_indices, scaler


def train_loop(
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    adult_weight: float = ADULT_CLASS_WEIGHT,
    seed: int = SEED,
) -> tuple[EmbeddingMLP, BinaryHead]:
    """Train embedding + binary head jointly. BCEWithLogits with adult class weight."""
    set_seed(seed)
    embedding = EmbeddingMLP().to(device)
    head = BinaryHead().to(device)
    optimizer = torch.optim.Adam(
        list(embedding.parameters()) + list(head.parameters()),
        lr=lr,
    )
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    X_t = torch.from_numpy(X).float().to(device)
    y_t = torch.from_numpy(y).float().unsqueeze(1).to(device)
    n = X_t.size(0)
    weights = torch.where(y_t == 1, 1.0, adult_weight).to(device)

    embedding.train()
    head.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        count = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb = X_t[idx]
            yb = y_t[idx]
            wb = weights[idx]
            optimizer.zero_grad()
            emb = embedding(xb)
            logits = head(emb)
            loss_per = criterion(logits.unsqueeze(1), yb)
            loss = (wb * loss_per).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            count += xb.size(0)
        avg = total_loss / count if count else 0.0
        logger.info(f"Epoch {epoch + 1}/{epochs} train_loss={avg:.4f}")
    return embedding, head


def compute_centroids(embeddings: np.ndarray, cluster_indices: np.ndarray) -> np.ndarray:
    """Label-based centroids for C1–C5. Shape (5, embedding_dim)."""
    centroids = np.zeros((5, embeddings.shape[1]), dtype=np.float32)
    for k in range(5):
        mask = cluster_indices == k
        if mask.sum() == 0:
            centroids[k] = embeddings.mean(axis=0)
            continue
        centroids[k] = embeddings[mask].mean(axis=0)
    return centroids


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: Train embedding + binary head on A; save centroids.")
    parser.add_argument("--train-manifest", type=Path, required=True, help="Path to train_manifest.csv (A)")
    parser.add_argument("--artifact-dir", type=Path, required=True, help="Directory for embedding, head, scaler, centroids")
    parser.add_argument("--scaler-path", type=Path, default=None, help="Path to scaler .npz; if omitted, fit on A and save to artifact-dir")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    X, y_binary, cluster_idx, scaler = load_a_features_and_labels(
        args.train_manifest,
        args.scaler_path,
        args.artifact_dir,
    )
    logger.info(f"A: {X.shape[0]} samples, binary minor={y_binary.sum():.0f} adult={(1-y_binary).sum():.0f}")

    embedding, head = train_loop(
        X, y_binary,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        adult_weight=ADULT_CLASS_WEIGHT,
        seed=args.seed,
    )

    embedding.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X).float().to(device)
        emb = embedding(X_t)
        emb_np = emb.cpu().numpy()

    centroids = compute_centroids(emb_np, cluster_idx)
    np.save(args.artifact_dir / "cluster_centroids.npy", centroids)
    logger.info(f"Saved cluster_centroids.npy shape {centroids.shape}")

    torch.save(embedding.state_dict(), args.artifact_dir / "embedding_weights.pt")
    torch.save(head.state_dict(), args.artifact_dir / "binary_head_weights.pt")
    logger.info("Saved embedding_weights.pt and binary_head_weights.pt")


if __name__ == "__main__":
    main()

"""
Evaluation utilities for baseline models.
Computes Accuracy (threshold-based), MAE (regression), F1-score and saves results.
"""
import json
import csv
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import RESULTS_DIR


def compute_baseline_metrics(
    age_true: np.ndarray,
    age_pred: np.ndarray,
    is_adult_true: np.ndarray,
    is_adult_pred: np.ndarray,
    age_group_true: Optional[np.ndarray] = None,
    age_group_pred: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute metrics for baseline comparison.
    
    Args:
        age_true: Ground truth age (continuous).
        age_pred: Predicted age (continuous).
        is_adult_true: Ground truth binary 18+ (0/1).
        is_adult_pred: Predicted binary 18+ (0/1).
        age_group_true: Optional ground truth age group indices (0-3).
        age_group_pred: Optional predicted age group indices.
    
    Returns:
        Dict with accuracy, mae, f1_binary (18+), and optionally f1_macro (age groups).
    """
    age_true = np.asarray(age_true).ravel()
    age_pred = np.asarray(age_pred).ravel()
    is_adult_true = np.asarray(is_adult_true).ravel().astype(int)
    is_adult_pred = np.asarray(is_adult_pred).ravel().astype(int)

    # Clamp predictions to valid range
    age_pred = np.clip(age_pred, 0, 100)

    mae = float(np.mean(np.abs(age_true - age_pred)))
    acc_binary = accuracy_score(is_adult_true, is_adult_pred)
    f1_binary = float(f1_score(is_adult_true, is_adult_pred, zero_division=0))

    metrics = {
        "accuracy_18": acc_binary,
        "mae": mae,
        "f1_18": f1_binary,
    }

    if age_group_true is not None and age_group_pred is not None:
        age_group_true = np.asarray(age_group_true).ravel().astype(int)
        age_group_pred = np.asarray(age_group_pred).ravel().astype(int)
        # Clip to 0-3 for 4 classes
        age_group_pred = np.clip(age_group_pred, 0, 3)
        metrics["accuracy_age_group"] = float(accuracy_score(age_group_true, age_group_pred))
        metrics["f1_macro_age_group"] = float(f1_score(age_group_true, age_group_pred, average="macro", zero_division=0))

    return metrics


def save_baseline_results(
    metrics: Dict[str, float],
    model_type: str,
    modality: str,
    output_dir: Optional[Path] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save baseline results in a comparison-friendly format (JSON and CSV row).
    
    Args:
        metrics: Dict from compute_baseline_metrics.
        model_type: e.g. "unimodal_face", "unimodal_audio", "late_fusion_baseline".
        modality: e.g. "face", "audio", "multimodal".
        output_dir: Directory to save (default: RESULTS_DIR).
        extra: Optional extra keys to include in JSON.
    
    Returns:
        Path to the saved JSON file.
    """
    output_dir = output_dir or RESULTS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_type": model_type,
        "modality": modality,
        **metrics,
    }
    if extra:
        payload["extra"] = extra

    json_path = output_dir / f"baseline_{model_type}.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    # Append one row to a comparison CSV if it exists or create it
    csv_path = output_dir / "baseline_comparison.csv"
    row = {"model_type": model_type, "modality": modality, **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}}
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return json_path


@torch.no_grad()
def evaluate_face_baseline(model, data_loader, device):
    """Run face baseline on a DataLoader; return lists of age_true, age_pred, is_adult_true, is_adult_pred, age_group_true, age_group_pred."""
    model.eval()
    age_true, age_pred = [], []
    is_adult_true, is_adult_pred = [], []
    age_group_true, age_group_pred = [], []
    for batch in data_loader:
        images = batch["image"].to(device)
        out = model(images)
        age_true.append(batch["age"].numpy())
        age_pred.append(out["age"].cpu().numpy())
        is_adult_true.append(batch["is_adult"].numpy())
        is_adult_pred.append((out["is_adult"].cpu().numpy() > 0.5).astype(np.float32))
        age_group_true.append(batch["age_group"].numpy())
        age_group_pred.append(out["age_group_logits"].argmax(dim=1).cpu().numpy())
    return (
        np.concatenate(age_true),
        np.concatenate(age_pred),
        np.concatenate(is_adult_true),
        np.concatenate(is_adult_pred),
        np.concatenate(age_group_true),
        np.concatenate(age_group_pred),
    )


@torch.no_grad()
def evaluate_audio_baseline(model, data_loader, device):
    """Run audio baseline on a DataLoader; return same structure as evaluate_face_baseline."""
    model.eval()
    age_true, age_pred = [], []
    is_adult_true, is_adult_pred = [], []
    age_group_true, age_group_pred = [], []
    for batch in data_loader:
        features = batch["features"].to(device)
        out = model(features)
        age_true.append(batch["age"].numpy())
        age_pred.append(out["age"].cpu().numpy())
        is_adult_true.append(batch["is_adult"].numpy())
        is_adult_pred.append((out["is_adult"].cpu().numpy() > 0.5).astype(np.float32))
        age_group_true.append(batch["age_group"].numpy())
        age_group_pred.append(out["age_group_logits"].argmax(dim=1).cpu().numpy())
    return (
        np.concatenate(age_true),
        np.concatenate(age_pred),
        np.concatenate(is_adult_true),
        np.concatenate(is_adult_pred),
        np.concatenate(age_group_true),
        np.concatenate(age_group_pred),
    )


@torch.no_grad()
def evaluate_fusion_baseline(model, data_loader, device):
    """Run fusion baseline on a DataLoader; same return structure."""
    model.eval()
    age_true, age_pred = [], []
    is_adult_true, is_adult_pred = [], []
    age_group_true, age_group_pred = [], []
    for batch in data_loader:
        face = batch["face"].to(device)
        voice = batch["voice"].to(device)
        out = model(face, voice)
        age_true.append(batch["age"].numpy())
        age_pred.append(out["age"].cpu().numpy())
        is_adult_true.append(batch["is_adult"].numpy())
        is_adult_pred.append((out["is_adult"].cpu().numpy() > 0.5).astype(np.float32))
        age_group_true.append(batch["age_group"].numpy())
        age_group_pred.append(out["age_group_logits"].argmax(dim=1).cpu().numpy())
    return (
        np.concatenate(age_true),
        np.concatenate(age_pred),
        np.concatenate(is_adult_true),
        np.concatenate(is_adult_pred),
        np.concatenate(age_group_true),
        np.concatenate(age_group_pred),
    )

"""
Train only the prediction head on top of frozen ImageBind (existing pretrained multimodal).

ImageBind was NOT trained for age prediction; no age supervision in pretraining.
Results are for comparison and limitation analysis only.
DO NOT retrain or fine-tune the ImageBind encoder; encoder remains frozen.
"""
import argparse
import random
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RANDOM_SEED, MODELS_DIR, DEVICE
from src.data.fusion_dataset import get_fusion_dataloaders
from src.training.eval_utils import compute_baseline_metrics


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train head on frozen ImageBind (existing pretrained multimodal)")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    set_seed(RANDOM_SEED)

    try:
        from src.models.external_multimodal.imagebind_wrapper import create_imagebind_age_model
    except ImportError as e:
        print("ImageBind not available:", e)
        print("Install from: https://github.com/facebookresearch/ImageBind")
        return

    loaders = get_fusion_dataloaders(batch_size=args.batch_size, num_workers=0, cache_only=False)
    model = create_imagebind_age_model(pretrained=True, freeze_encoder=True, device=DEVICE)

    # Optimizer ONLY on head parameters
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(head_params, lr=args.lr, weight_decay=1e-5)
    mae_fn = torch.nn.L1Loss()
    ce_fn = torch.nn.CrossEntropyLoss()
    bce_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n = 0
        pbar = tqdm(loaders["train"], desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for batch in pbar:
            face_paths = batch["face_filepath"]
            audio_paths = batch["audio_filepath"]
            age = batch["age"].to(DEVICE)
            age_group = batch["age_group"].to(DEVICE)
            is_adult = batch["is_adult"].to(DEVICE)

            optimizer.zero_grad()
            out = model(image_paths=face_paths, audio_paths=audio_paths, device=DEVICE)
            loss = mae_fn(out["age"], age) + 0.5 * ce_fn(out["age_group_logits"], age_group) + 0.5 * bce_fn(out["is_adult_logit"], is_adult)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * age.size(0)
            n += age.size(0)
            pbar.set_postfix(loss=loss.item())

    # Evaluation (same logic as baselines)
    model.eval()
    age_t, age_p, isadult_t, isadult_p = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loaders["test"], desc="Evaluating"):
            out = model(image_paths=batch["face_filepath"], audio_paths=batch["audio_filepath"], device=DEVICE)
            age_t.append(batch["age"].numpy())
            age_p.append(out["age"].cpu().numpy())
            isadult_t.append(batch["is_adult"].numpy())
            isadult_p.append((out["is_adult"].cpu().numpy() > 0.5).astype(np.float32))

    age_t = np.concatenate(age_t)
    age_p = np.clip(np.concatenate(age_p), 0, 100)
    isadult_t = np.concatenate(isadult_t).astype(int)
    isadult_p = np.concatenate(isadult_p).astype(int)
    metrics = compute_baseline_metrics(age_t, age_p, isadult_t, isadult_p)

    # Result reporting: Existing Pretrained Multimodal
    print()
    print("=== EXISTING PRETRAINED MULTIMODAL RESULTS ===")
    print()
    print("| Model | Type | Accuracy (18+) | MAE | F1 |")
    print("|-------|------|----------------|-----|-----|")
    acc_pct = metrics["accuracy_18"] * 100
    print(f"| ImageBind + Head | Existing Multimodal | {acc_pct:.1f}% | {metrics['mae']:.2f} | {metrics['f1_18']:.2f} |")
    print()
    print("(ImageBind was NOT trained for age prediction; results for comparison and limitation analysis only.)")


if __name__ == "__main__":
    main()

"""
Train only the prediction head on top of frozen CLAP (existing pretrained multimodal, audio).

CLAP was NOT trained for age prediction; no age supervision in pretraining.
Results are for comparison and limitation analysis only.
DO NOT retrain or fine-tune the CLAP encoder; encoder remains frozen.
"""
print("RUNNING FILE:", __file__)

import argparse
import random
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RANDOM_SEED, DEVICE
from src.data.data_analysis import DataAnalyzer
from src.data.audio_dataset import get_audio_dataloaders
from src.training.eval_utils import compute_baseline_metrics


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train head on frozen CLAP (existing pretrained multimodal, audio)")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-batches", type=int, default=200, help="Limit number of training batches per epoch")
    args = parser.parse_args()
    print("MAX_BATCHES =", args.max_batches)

    MAX_BATCHES = args.max_batches
    MAX_EVAL_BATCHES = 200

    set_seed(RANDOM_SEED)

    try:
        from src.models.external_multimodal.clap_wrapper import create_clap_age_model
    except ImportError as e:
        print("CLAP not available:", e)
        print("pip install transformers librosa")
        return

    analyzer = DataAnalyzer()
    audio_data = analyzer.prepare_audio_data()
    loaders = get_audio_dataloaders(audio_data, batch_size=args.batch_size, num_workers=0, cache_only=False)
    if len(loaders["train"].dataset) == 0:
        print("No cached audio samples; cannot train CLAP head. Precompute cache first.")
        return

    model = create_clap_age_model(freeze_encoder=True, device=DEVICE)

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

        pbar = tqdm(loaders["train"], total=MAX_BATCHES, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)


        for i, batch in enumerate(pbar):
            if i >= MAX_BATCHES:
                break   # ⬅️ THIS limits to 200 batches

            audio_paths = batch["filepath"]
            age = batch["age"].to(DEVICE)
            age_group = batch["age_group"].to(DEVICE)
            is_adult = batch["is_adult"].to(DEVICE)

            optimizer.zero_grad()
            out = model(audio_paths=audio_paths, device=DEVICE)

            loss = (
                mae_fn(out["age"], age)
                + 0.5 * ce_fn(out["age_group_logits"], age_group)
                + 0.5 * bce_fn(out["is_adult_logit"], is_adult)
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * age.size(0)
            n += age.size(0)

            pbar.set_postfix(loss=f"{loss.item():.2f}")


    # Evaluation (same as baselines)
    model.eval()
    age_t, age_p, isadult_t, isadult_p = [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loaders["test"], desc="Evaluating")):
            if i >= MAX_EVAL_BATCHES:
                break
            out = model(audio_paths=batch["filepath"], device=DEVICE)
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
    print(f"| CLAP + Head | Existing Pretrained | {acc_pct:.1f}% | {metrics['mae']:.2f} | {metrics['f1_18']:.2f} |")
    print()
    print("(CLAP was NOT trained for age prediction; results for comparison and limitation analysis only.)")


if __name__ == "__main__":
    main()

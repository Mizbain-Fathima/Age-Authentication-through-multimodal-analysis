"""
End-to-end training for the single joint multimodal model.
"""
import argparse
import random
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RANDOM_SEED, DEVICE, MODELS_DIR
from src.data.data_analysis import DataAnalyzer
from src.data.multimodal_dataset import get_multimodal_dataloaders
from src.models.multimodal_model import create_multimodal_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train single joint multimodal model")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    set_seed(RANDOM_SEED)

    analyzer = DataAnalyzer()
    face_data = analyzer.prepare_face_data()
    audio_data = analyzer.prepare_audio_data()

    loaders = get_multimodal_dataloaders(
        face_data, audio_data,
        batch_size=args.batch_size,
        num_workers=0,
        seed=RANDOM_SEED,
    )
    audio_dim = loaders["audio_dim"]

    model = create_multimodal_model(audio_input_dim=audio_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    mae_fn = torch.nn.L1Loss()
    ce_fn = torch.nn.CrossEntropyLoss()
    bce_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n = 0
        pbar = tqdm(loaders["train"], desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for batch in pbar:
            image = batch["image"].to(DEVICE)
            audio = batch["audio"].to(DEVICE)
            age = batch["age"].to(DEVICE)
            age_group = batch["age_group"].to(DEVICE)
            is_adult = batch["is_adult"].to(DEVICE)

            optimizer.zero_grad()
            out = model(image, audio)
            loss = (
                mae_fn(out["age"], age)
                + 0.5 * ce_fn(out["age_group_logits"], age_group)
                + 0.5 * bce_fn(out["is_adult_logit"], is_adult)
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * age.size(0)
            n += age.size(0)
            pbar.set_postfix(loss=loss.item())
        avg_loss = total_loss / n if n else 0.0
        print(f"Epoch {epoch+1}/{args.epochs}  train_loss: {avg_loss:.4f}")

    model.eval()
    age_true, age_pred, isadult_true, isadult_pred = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loaders["test"], desc="Evaluating"):
            out = model(
                batch["image"].to(DEVICE),
                batch["audio"].to(DEVICE),
            )
            age_true.append(batch["age"].numpy())
            age_pred.append(out["age"].cpu().numpy())
            isadult_true.append(batch["is_adult"].numpy())
            isadult_pred.append((out["is_adult_logit"].sigmoid().cpu().numpy() > 0.5).astype(np.float32))

    age_true = np.concatenate(age_true)
    age_pred = np.clip(np.concatenate(age_pred), 0, 100)
    isadult_true = np.concatenate(isadult_true).astype(int)
    isadult_pred = np.concatenate(isadult_pred).astype(int)

    mae = float(np.mean(np.abs(age_true - age_pred)))
    acc = float(accuracy_score(isadult_true, isadult_pred))
    f1 = float(f1_score(isadult_true, isadult_pred, zero_division=0))

    print()
    print("| Model | Type | Accuracy (18+) | MAE | F1 |")
    print("|-------|------|----------------|-----|-----|")
    print(f"| Single Multimodal (Ours) | Joint end-to-end | {acc*100:.1f}% | {mae:.2f} | {f1:.2f} |")

    ckpt_path = MODELS_DIR / "multimodal_age_model_best.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": args.epochs,
        "audio_dim": audio_dim,
    }, ckpt_path)
    print(f"\nCheckpoint saved: {ckpt_path}")


if __name__ == "__main__":
    main()

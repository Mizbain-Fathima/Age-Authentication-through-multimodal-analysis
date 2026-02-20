"""
End-to-end training for the fusion multimodal model (face + voice encoders, joint heads).
Supports full joint training, optional init from unimodal checkpoints, and ablations.
"""
import argparse
import random
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RANDOM_SEED, DEVICE, MODELS_DIR, FUSION_MODELS_DIR
from src.data.data_analysis import DataAnalyzer
from src.data.multimodal_dataset import get_multimodal_dataloaders
from src.models.fusion_model import create_fusion_model, MultimodalFusionModel
from src.models.face_model import load_face_model
from src.models.voice_model import load_voice_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _init_fusion_from_unimodal(model: MultimodalFusionModel, face_ckpt: str, voice_ckpt: str):
    if face_ckpt and Path(face_ckpt).exists():
        face_model = load_face_model(face_ckpt)
        model.face_encoder[0].load_state_dict(face_model.backbone.state_dict(), strict=True)
        model.face_encoder[1].load_state_dict(face_model.feature_processor.state_dict(), strict=True)
        del face_model
    if voice_ckpt and Path(voice_ckpt).exists():
        voice_model = load_voice_model(voice_ckpt)
        model.voice_encoder.load_state_dict(voice_model.state_dict(), strict=False)


def train_multimodal_model(
    resume=False,
    init_from_unimodal=False,
    epochs=None,
    batch_size=None,
    lr=None,
    ablation_mode=None,
):
    """Train the fusion multimodal model end-to-end.

    Can be called programmatically (e.g. from run.py) or via the CLI (main()).

    Args:
        resume: Reserved for future checkpoint-resume support.
        init_from_unimodal: If True, initialise face/voice encoders from pretrained checkpoints.
        epochs: Number of training epochs.  Falls back to the hardcoded default (12) when None.
        batch_size: Batch size.  Falls back to the hardcoded default (16) when None.
        lr: Learning rate.  Falls back to the hardcoded default (1e-4) when None.
        ablation_mode: One of 'face_only', 'voice_only', 'no_fusion', or None (full fusion).
    """
    # Apply defaults
    _epochs = epochs if epochs is not None else 12
    _batch_size = batch_size if batch_size is not None else 16
    _lr = lr if lr is not None else 1e-4

    set_seed(RANDOM_SEED)

    analyzer = DataAnalyzer()
    face_data = analyzer.prepare_face_data()
    audio_data = analyzer.prepare_audio_data()

    loaders = get_multimodal_dataloaders(
        face_data, audio_data,
        batch_size=_batch_size,
        num_workers=0,
        seed=RANDOM_SEED,
    )
    audio_dim = loaders["audio_dim"]

    model = create_fusion_model(audio_input_dim=audio_dim, freeze_encoders=False)
    if init_from_unimodal:
        face_ckpt = MODELS_DIR / "face_age_model_best.pth"
        voice_ckpt = MODELS_DIR / "voice_age_model_best.pth"
        _init_fusion_from_unimodal(model, str(face_ckpt), str(voice_ckpt))

    optimizer = torch.optim.AdamW(model.parameters(), lr=_lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_epochs)
    mae_fn = torch.nn.L1Loss()
    ce_fn = torch.nn.CrossEntropyLoss()
    bce_fn = torch.nn.BCEWithLogitsLoss()

    best_val_mae = float("inf")
    best_state = None

    for epoch in range(_epochs):
        model.train()
        total_loss = 0.0
        n = 0
        pbar = tqdm(loaders["train"], desc=f"Epoch {epoch+1}/{_epochs}", leave=False)
        for batch in pbar:
            image = batch["image"].to(DEVICE)
            audio = batch["audio"].to(DEVICE)
            age = batch["age"].to(DEVICE)
            age_group = batch["age_group"].to(DEVICE)
            is_adult = batch["is_adult"].to(DEVICE)

            optimizer.zero_grad()
            out = model(image, audio, ablation=ablation_mode)
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
        scheduler.step()
        avg_loss = total_loss / n if n else 0.0
        print(f"Epoch {epoch+1}/{_epochs}  train_loss: {avg_loss:.4f}")

        model.eval()
        val_mae_sum = 0.0
        val_n = 0
        val_correct_adult = 0
        val_ages_list = []
        val_pred_ages_list = []
        val_is_adult_list = []
        val_pred_adult_list = []
        with torch.no_grad():
            for batch in loaders["val"]:
                image = batch["image"].to(DEVICE)
                audio = batch["audio"].to(DEVICE)
                age = batch["age"]
                is_adult = batch["is_adult"]
                out = model(image, audio, ablation=ablation_mode)
                pred_age = out["age"].cpu()
                pred_adult = (out["is_adult_logit"].sigmoid().cpu() > 0.5).float()
                val_mae_sum += torch.abs(pred_age - age).sum().item()
                val_n += age.size(0)
                val_correct_adult += (pred_adult == is_adult).sum().item()
                val_ages_list.append(age.numpy())
                val_pred_ages_list.append(pred_age.numpy())
                val_is_adult_list.append(is_adult.numpy())
                val_pred_adult_list.append(pred_adult.numpy())
        val_mae = val_mae_sum / val_n if val_n else float("inf")
        val_acc = val_correct_adult / val_n if val_n else 0.0
        print(f"  val_MAE: {val_mae:.4f}  val_adult_acc: {val_acc:.4f}")
        if val_ages_list and val_pred_ages_list:
            val_ages_all = np.concatenate(val_ages_list)
            val_pred_ages_all = np.concatenate(val_pred_ages_list)
            mask_15_21 = (val_ages_all >= 15) & (val_ages_all <= 21)
            if np.any(mask_15_21):
                val_mae_15_21 = float(np.mean(np.abs(val_ages_all[mask_15_21] - val_pred_ages_all[mask_15_21])))
                print(f"  val_MAE(age 15-21): {val_mae_15_21:.4f}")
        if val_is_adult_list and val_pred_adult_list:
            y_true = np.concatenate(val_is_adult_list).astype(int)
            y_pred = np.concatenate(val_pred_adult_list).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            print(f"  val_adult_confusion: {cm.tolist()}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  -> new best val MAE")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    age_true, age_pred, isadult_true, isadult_pred = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loaders["test"], desc="Evaluating"):
            out = model(
                batch["image"].to(DEVICE),
                batch["audio"].to(DEVICE),
                ablation=ablation_mode,
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
    print(f"| Fusion (Ours) | Joint end-to-end | {acc*100:.1f}% | {mae:.2f} | {f1:.2f} |")

    ckpt_path = FUSION_MODELS_DIR / "fusion_age_model_best.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": _epochs,
        "audio_dim": audio_dim,
        "model_type": "fusion",
    }, ckpt_path)
    print(f"\nFusion checkpoint saved: {ckpt_path}")


def main():
    parser = argparse.ArgumentParser(description="Train fusion multimodal model (end-to-end)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--init-from-unimodal", action="store_true", help="Load pretrained face/voice and continue joint training")
    parser.add_argument("--ablation-face-only", action="store_true", help="Train with face encoder only (voice zeroed)")
    parser.add_argument("--ablation-voice-only", action="store_true", help="Train with voice encoder only (face zeroed)")
    parser.add_argument("--ablation-no-fusion", action="store_true", help="Concat features without gated fusion block")
    args = parser.parse_args()

    ablation_mode = None
    if args.ablation_face_only:
        ablation_mode = "face_only"
    elif args.ablation_voice_only:
        ablation_mode = "voice_only"
    elif args.ablation_no_fusion:
        ablation_mode = "no_fusion"

    train_multimodal_model(
        init_from_unimodal=args.init_from_unimodal,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        ablation_mode=ablation_mode,
    )


if __name__ == "__main__":
    main()


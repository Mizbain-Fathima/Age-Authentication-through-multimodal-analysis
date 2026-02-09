# eval_baselines.py
# Evaluation-only script for baseline models
# NO training, NO file writes, NO code modification elsewhere

import torch
import numpy as np
from sklearn.metrics import f1_score

from src.config import DEVICE, MODELS_DIR
from src.data.data_analysis import DataAnalyzer
from src.data.face_dataset import get_face_dataloaders
from src.data.audio_dataset import get_audio_dataloaders
from src.data.fusion_dataset import get_fusion_dataloaders

from src.models.face_model import create_face_model
from src.models.voice_model import create_voice_model
from src.models.fusion_model import create_fusion_model


def eval_face_model(model, loader):
    model.eval()
    y_true_age, y_pred_age = [], []
    y_true_adult, y_pred_adult = [], []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(DEVICE)
            targets = batch["age"].to(DEVICE)

            outputs = model(images)
            preds = outputs["age"]

            y_true_age.extend(targets.cpu().numpy())
            y_pred_age.extend(preds.cpu().numpy())

            y_true_adult.extend((targets >= 18).cpu().numpy())
            y_pred_adult.extend((preds >= 18).cpu().numpy())

    mae = np.mean(np.abs(np.array(y_true_age) - np.array(y_pred_age)))
    acc = np.mean(np.array(y_true_adult) == np.array(y_pred_adult))
    f1 = f1_score(y_true_adult, y_pred_adult, zero_division=0)

    return acc, mae, f1


def eval_voice_model(model, loader):
    model.eval()
    y_true_age, y_pred_age = [], []
    y_true_adult, y_pred_adult = [], []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(DEVICE)
            targets = batch["age"].to(DEVICE)

            outputs = model(features)
            preds = outputs["age"]

            y_true_age.extend(targets.cpu().numpy())
            y_pred_age.extend(preds.cpu().numpy())

            y_true_adult.extend((targets >= 18).cpu().numpy())
            y_pred_adult.extend((preds >= 18).cpu().numpy())

    mae = np.mean(np.abs(np.array(y_true_age) - np.array(y_pred_age)))
    acc = np.mean(np.array(y_true_adult) == np.array(y_pred_adult))
    f1 = f1_score(y_true_adult, y_pred_adult, zero_division=0)

    return acc, mae, f1


def eval_fusion_model(model, loader):
    model.eval()
    y_true_age, y_pred_age = [], []
    y_true_adult, y_pred_adult = [], []

    with torch.no_grad():
        for batch in loader:
            face = batch["face"].to(DEVICE)
            voice = batch["voice"].to(DEVICE)
            targets = batch["age"].to(DEVICE)

            outputs = model(face, voice)
            preds = outputs["age"]

            y_true_age.extend(targets.cpu().numpy())
            y_pred_age.extend(preds.cpu().numpy())

            y_true_adult.extend((targets >= 18).cpu().numpy())
            y_pred_adult.extend((preds >= 18).cpu().numpy())

    mae = np.mean(np.abs(np.array(y_true_age) - np.array(y_pred_age)))
    acc = np.mean(np.array(y_true_adult) == np.array(y_pred_adult))
    f1 = f1_score(y_true_adult, y_pred_adult, zero_division=0)

    return acc, mae, f1


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    analyzer = DataAnalyzer()

    # ================= FACE BASELINE =================
    face_model = create_face_model(
        backbone="efficientnet_b0",
        pretrained=False
    )

    face_ckpt = torch.load(
        MODELS_DIR / "face_age_model_best.pth",
        map_location=DEVICE
    )
    face_model.load_state_dict(face_ckpt["model_state_dict"])
    face_model.to(DEVICE)

    face_data = analyzer.prepare_face_data()
    face_loaders = get_face_dataloaders(
        face_data,
        batch_size=16,
        num_workers=0  # Windows-safe
    )
    face_loader = face_loaders["test"]

    face_acc, face_mae, face_f1 = eval_face_model(face_model, face_loader)

    # ================= AUDIO BASELINE =================
    voice_model = create_voice_model()

    voice_ckpt = torch.load(
        MODELS_DIR / "voice_age_model_best.pth",
        map_location=DEVICE
    )
    voice_model.load_state_dict(voice_ckpt["model_state_dict"])
    voice_model.to(DEVICE)

    audio_data = analyzer.prepare_audio_data()
    audio_loaders = get_audio_dataloaders(
        audio_data,
        batch_size=16,
        num_workers=0,   # Windows-safe
        cache_only=True
    )
    audio_loader = audio_loaders["test"]

    audio_acc, audio_mae, audio_f1 = eval_voice_model(voice_model, audio_loader)

    # ================= FUSION BASELINE =================
    fusion_model = create_fusion_model(
        face_checkpoint=str(MODELS_DIR / "face_age_model_best.pth"),
        voice_checkpoint=str(MODELS_DIR / "voice_age_model_best.pth"),
        freeze_encoders=True
    )

    fusion_ckpt = torch.load(
        MODELS_DIR / "fusion_age_model_best.pth",
        map_location=DEVICE
    )
    fusion_model.load_state_dict(fusion_ckpt["model_state_dict"])
    fusion_model.to(DEVICE)

    fusion_loaders = get_fusion_dataloaders(
        batch_size=16,
        num_workers=0,   # Windows-safe
        cache_only=False
    )
    fusion_loader = fusion_loaders["test"]

    fusion_acc, fusion_mae, fusion_f1 = eval_fusion_model(fusion_model, fusion_loader)

    # ================= RESULTS =================
    print("\n=== BASELINE COMPARISON RESULTS ===")
    print(f"Face-only   | Acc: {face_acc*100:.1f}% | MAE: {face_mae:.2f} | F1: {face_f1:.2f}")
    print(f"Audio-only  | Acc: {audio_acc*100:.1f}% | MAE: {audio_mae:.2f} | F1: {audio_f1:.2f}")
    print(f"Late Fusion | Acc: {fusion_acc*100:.1f}% | MAE: {fusion_mae:.2f} | F1: {fusion_f1:.2f}")

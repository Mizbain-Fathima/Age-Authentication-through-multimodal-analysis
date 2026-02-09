"""
Late Fusion Multimodal Baseline — STANDARD baseline for comparison.

This model is a BASELINE used for comparison against a proposed
joint end-to-end multimodal age authentication model.

- Pretrained face encoder + audio encoder (existing architectures).
- Fusion: concatenation or weighted average of embeddings.
- Shallow MLP classifier/regressor on top.
- No cross-attention; no joint backpropagation across encoders.
Saves: model checkpoints, evaluation metrics (Accuracy, MAE, F1).
"""
import argparse
import random
import numpy as np
import torch
from pathlib import Path

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

PROJECT_ROOT = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RANDOM_SEED, MODELS_DIR
from src.data.fusion_dataset import get_fusion_dataloaders
from src.models.late_fusion_baseline import create_late_fusion_baseline
from src.training.trainer import FusionTrainer
from src.training.eval_utils import (
    compute_baseline_metrics,
    save_baseline_results,
    evaluate_fusion_baseline,
)
from loguru import logger


class LateFusionLoss(torch.nn.Module):
    """Loss for Late Fusion Baseline (same targets as fusion: age, age_group, is_adult)."""

    def __init__(self):
        super().__init__()
        self.mae = torch.nn.L1Loss()
        self.mse = torch.nn.MSELoss()
        self.ce = torch.nn.CrossEntropyLoss()
        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, predictions, targets):
        age_loss = self.mae(predictions["age"], targets["age"]) + 0.5 * self.mse(predictions["age"], targets["age"])
        group_loss = self.ce(predictions["age_group_logits"], targets["age_group"])
        adult_loss = self.bce(predictions["is_adult_logit"], targets["is_adult"])
        total = age_loss + 0.5 * group_loss + 0.5 * adult_loss
        return {"total": total, "age": age_loss, "group": group_loss, "adult": adult_loss}


def main():
    parser = argparse.ArgumentParser(description="Train Late Fusion Baseline (multimodal)")
    parser.add_argument("--fusion", default="concat", choices=["concat", "weighted_avg"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--resume", action="store_true", help="Resume from best checkpoint")
    args = parser.parse_args()

    set_seed(RANDOM_SEED)

    # Use cached audio embeddings only (no recomputation)
    loaders = get_fusion_dataloaders(batch_size=args.batch_size, num_workers=0, cache_only=True)

    face_ckpt = MODELS_DIR / "baseline_face_best.pth"
    voice_ckpt = MODELS_DIR / "baseline_audio_best.pth"
    if not face_ckpt.exists():
        face_ckpt = MODELS_DIR / "face_age_model_best.pth"
    if not voice_ckpt.exists():
        voice_ckpt = MODELS_DIR / "voice_age_model_best.pth"

    model = create_late_fusion_baseline(
        face_checkpoint=str(face_ckpt) if face_ckpt.exists() else None,
        voice_checkpoint=str(voice_ckpt) if voice_ckpt.exists() else None,
        fusion_method=args.fusion,
        freeze_encoders=True,
    )
    criterion = LateFusionLoss()

    trainer = FusionTrainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        criterion=criterion,
        epochs=args.epochs,
        model_name="baseline_late_fusion",
        use_amp=False,
    )

    start_epoch = 0
    if args.resume:
        ckpt = MODELS_DIR / "baseline_late_fusion_best.pth"
        if ckpt.exists():
            start_epoch = trainer.load_checkpoint("baseline_late_fusion_best.pth")
        else:
            logger.warning("No checkpoint found, starting from scratch")

    trainer.train(save_best=True, early_stopping=3, start_epoch=start_epoch)

    at, ap, it, ip, gt, gp = evaluate_fusion_baseline(trainer.model, loaders["test"], trainer.device)
    metrics = compute_baseline_metrics(at, ap, it, ip, gt, gp)
    out_path = save_baseline_results(
        metrics,
        model_type="late_fusion_baseline",
        modality="multimodal",
        extra={"fusion_method": args.fusion},
    )
    logger.info(f"Late Fusion Baseline metrics: {metrics}")
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()

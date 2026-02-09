"""
Unimodal Face Baseline — STANDARD baseline for comparison.

This model is a BASELINE used for comparison against a proposed
joint end-to-end multimodal age authentication model.

Architecture: pretrained ResNet-50 or EfficientNet (existing in repo).
Input: UTKFace images. Output: age regression and age-threshold classification (18+).
Saves: model checkpoints, evaluation metrics (Accuracy, MAE, F1).
"""
import argparse
import random
import numpy as np
import torch
from pathlib import Path

# Set seeds for reproducibility (do not hardcode dataset paths; use config)
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RANDOM_SEED, MODELS_DIR, RESULTS_DIR
from src.data.data_analysis import DataAnalyzer
from src.data.face_dataset import get_face_dataloaders
from src.models.face_model import create_face_model, FaceAgeLoss
from src.training.trainer import FaceTrainer
from src.training.eval_utils import (
    compute_baseline_metrics,
    save_baseline_results,
    evaluate_face_baseline,
)
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Train unimodal face baseline (UTKFace)")
    parser.add_argument("--backbone", default="efficientnet_b0", choices=["efficientnet_b0", "resnet50"])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--resume", action="store_true", help="Resume from best checkpoint")
    args = parser.parse_args()

    set_seed(RANDOM_SEED)

    # Data from config paths (no hardcoding)
    analyzer = DataAnalyzer()
    face_data = analyzer.prepare_face_data()
    loaders = get_face_dataloaders(face_data, batch_size=args.batch_size, num_workers=0)

    # Unimodal face baseline: existing architecture
    model = create_face_model(backbone=args.backbone, pretrained=True)
    criterion = FaceAgeLoss()

    trainer = FaceTrainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        criterion=criterion,
        epochs=args.epochs,
        model_name="baseline_face",
        use_amp=False,
    )

    start_epoch = 0
    if args.resume:
        ckpt = MODELS_DIR / "baseline_face_best.pth"
        if ckpt.exists():
            start_epoch = trainer.load_checkpoint("baseline_face_best.pth")
        else:
            logger.warning("No checkpoint found, starting from scratch")

    trainer.train(save_best=True, early_stopping=5, start_epoch=start_epoch)

    # Evaluate on test set and save metrics
    at, ap, it, ip, gt, gp = evaluate_face_baseline(trainer.model, loaders["test"], trainer.device)
    metrics = compute_baseline_metrics(at, ap, it, ip, gt, gp)
    out_path = save_baseline_results(
        metrics,
        model_type="unimodal_face",
        modality="face",
        extra={"backbone": args.backbone},
    )
    logger.info(f"Face baseline metrics: {metrics}")
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()

"""
Unimodal Audio Baseline — STANDARD baseline for comparison.

This model is a BASELINE used for comparison against a proposed
joint end-to-end multimodal age authentication model.

Architecture: pretrained / existing voice model (MFCC + log-mel from cache).
Input: cached audio embeddings only (no recomputation).
Output: age regression and age-threshold classification (18+).
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
from src.data.data_analysis import DataAnalyzer
from src.data.audio_dataset import get_audio_dataloaders
from src.models.voice_model import create_voice_model, VoiceAgeLoss
from src.training.trainer import VoiceTrainer
from src.training.eval_utils import (
    compute_baseline_metrics,
    save_baseline_results,
    evaluate_audio_baseline,
)
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Train unimodal audio baseline (cached embeddings only)")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--resume", action="store_true", help="Resume from best checkpoint")
    args = parser.parse_args()

    set_seed(RANDOM_SEED)

    analyzer = DataAnalyzer()
    audio_data = analyzer.prepare_audio_data()
    # Load cached embeddings only; do not recompute audio features
    loaders = get_audio_dataloaders(
        audio_data,
        batch_size=args.batch_size,
        num_workers=0,
        cache_only=True,
    )

    model = create_voice_model()
    criterion = VoiceAgeLoss()

    trainer = VoiceTrainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        criterion=criterion,
        epochs=args.epochs,
        model_name="baseline_audio",
        use_amp=False,
    )

    start_epoch = 0
    if args.resume:
        ckpt = MODELS_DIR / "baseline_audio_best.pth"
        if ckpt.exists():
            start_epoch = trainer.load_checkpoint("baseline_audio_best.pth")
        else:
            logger.warning("No checkpoint found, starting from scratch")

    trainer.train(save_best=True, early_stopping=5, start_epoch=start_epoch)

    at, ap, it, ip, gt, gp = evaluate_audio_baseline(trainer.model, loaders["test"], trainer.device)
    metrics = compute_baseline_metrics(at, ap, it, ip, gt, gp)
    out_path = save_baseline_results(metrics, model_type="unimodal_audio", modality="audio")
    logger.info(f"Audio baseline metrics: {metrics}")
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()

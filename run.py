#!/usr/bin/env python
"""
Age Authentication System - Main Entry Point

Usage:
    python run.py server              # Start the API server
    python run.py train face          # Train face age model
    python run.py train voice         # Train voice age model
    python run.py train voice --resume # Resume voice training from checkpoint
    python run.py train fusion        # Train fusion model
    python run.py train fusion --resume
    python run.py train fusion --init-from-unimodal
    python run.py train fusion --epochs 25 --batch-size 16 --lr 1e-4
    python run.py train fusion --ablation-face-only
    python run.py analyze             # Run data analysis
"""
import sys
import argparse
from pathlib import Path
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def run_server():
    """Start the FastAPI server"""
    from src.api.main import run_server
    run_server()


def train_model(
    model_type,
    resume=False,
    init_from_unimodal=False,
    epochs=None,
    batch_size=None,
    lr=None,
    ablation_mode=None,
):
    """Train a specific model.

    Args:
        model_type: Type of model to train (face, voice, fusion)
        resume: If True, resume training from the last saved checkpoint
        init_from_unimodal: (fusion only) Initialise encoders from pretrained unimodal weights
        epochs: (fusion only) Override number of training epochs
        batch_size: (fusion only) Override batch size
        lr: (fusion only) Override learning rate
        ablation_mode: (fusion only) One of 'face_only', 'voice_only', 'no_fusion', or None
    """
    if model_type == 'face':
        from src.training.trainer import train_face_model
        if resume:
            logger.info("Resuming face model training from checkpoint...")
        else:
            logger.info("Starting face model training...")
        trainer, history = train_face_model(resume=resume)
        logger.info("Face model training complete!")

    elif model_type == 'voice':
        from src.training.trainer import train_voice_model
        if resume:
            logger.info("Resuming voice model training from checkpoint...")
        else:
            logger.info("Starting voice model training...")
        trainer, history = train_voice_model(resume=resume)
        logger.info("Voice model training complete!")

    elif model_type == 'fusion':
        from src.training.train_multimodal import train_multimodal_model
        logger.info("Starting fusion model training...")
        if init_from_unimodal:
            logger.info("Initialising encoders from pretrained unimodal checkpoints")
        train_multimodal_model(
            resume=resume,
            init_from_unimodal=init_from_unimodal,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            ablation_mode=ablation_mode,
        )
        logger.info("Fusion model training complete!")

    else:
        logger.error(f"Unknown model type: {model_type}")
        sys.exit(1)


def run_analysis():
    """Run data analysis"""
    from src.data.data_analysis import run_full_analysis
    
    logger.info("Running comprehensive data analysis...")
    analyzer, face_data, audio_data = run_full_analysis()
    logger.info("Data analysis complete!")
    
    return analyzer


def main():
    parser = argparse.ArgumentParser(
        description="Age Authentication System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py server              Start the API server on port 8000
  python run.py train face          Train the face age prediction model
  python run.py train voice         Train the voice age prediction model
  python run.py train voice --resume Resume voice model training from checkpoint
  python run.py train fusion        Train the fusion model (requires face+voice models)
  python run.py analyze             Analyze the datasets and show statistics
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Server command
    server_parser = subparsers.add_parser('server', help='Start the API server')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument(
        'model',
        choices=['face', 'voice', 'fusion'],
        help='Model type to train'
    )
    train_parser.add_argument(
        '--resume', '-r',
        action='store_true',
        help='Resume training from the last saved checkpoint'
    )

    # Fusion-only flags (ignored when model is face or voice)
    train_parser.add_argument(
        '--init-from-unimodal',
        action='store_true',
        default=False,
        help='(fusion) Initialise encoders from pretrained face/voice checkpoints'
    )
    train_parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        metavar='N',
        help='(fusion) Number of training epochs (overrides config default)'
    )
    train_parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        metavar='N',
        help='(fusion) Batch size (overrides config default)'
    )
    train_parser.add_argument(
        '--lr',
        type=float,
        default=None,
        metavar='LR',
        help='(fusion) Learning rate (overrides config default)'
    )
    _ablation = train_parser.add_mutually_exclusive_group()
    _ablation.add_argument(
        '--ablation-face-only',
        action='store_true',
        default=False,
        help='(fusion) Train with face encoder only'
    )
    _ablation.add_argument(
        '--ablation-voice-only',
        action='store_true',
        default=False,
        help='(fusion) Train with voice encoder only'
    )
    _ablation.add_argument(
        '--ablation-no-fusion',
        action='store_true',
        default=False,
        help='(fusion) Concatenate features without gated fusion block'
    )
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Run data analysis')
    
    args = parser.parse_args()
    
    if args.command == 'server':
        run_server()
    elif args.command == 'train':
        # Resolve fusion-only ablation flag into a single string (or None)
        ablation_mode = None
        if args.model == 'fusion':
            if args.ablation_face_only:
                ablation_mode = 'face_only'
            elif args.ablation_voice_only:
                ablation_mode = 'voice_only'
            elif args.ablation_no_fusion:
                ablation_mode = 'no_fusion'

        train_model(
            args.model,
            resume=args.resume,
            init_from_unimodal=args.init_from_unimodal if args.model == 'fusion' else False,
            epochs=args.epochs if args.model == 'fusion' else None,
            batch_size=args.batch_size if args.model == 'fusion' else None,
            lr=args.lr if args.model == 'fusion' else None,
            ablation_mode=ablation_mode,
        )
    elif args.command == 'analyze':
        run_analysis()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()


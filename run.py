#!/usr/bin/env python
"""
Age Authentication System - Main Entry Point

Usage:
    python run.py server              # Start the API server
    python run.py train face          # Train face age model
    python run.py train voice         # Train voice age model
    python run.py train voice --resume # Resume voice training from checkpoint
    python run.py train fusion        # Train fusion model
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


def train_model(model_type, resume=False):
    """Train a specific model
    
    Args:
        model_type: Type of model to train (face, voice, fusion)
        resume: If True, resume training from the last saved checkpoint
    """
    from src.training.trainer import train_face_model, train_voice_model, train_fusion_model
    
    if model_type == 'face':
        if resume:
            logger.info("Resuming face model training from checkpoint...")
        else:
            logger.info("Starting face model training...")
        trainer, history = train_face_model(resume=resume)
        logger.info("Face model training complete!")
        
    elif model_type == 'voice':
        if resume:
            logger.info("Resuming voice model training from checkpoint...")
        else:
            logger.info("Starting voice model training...")
        trainer, history = train_voice_model(resume=resume)
        logger.info("Voice model training complete!")
        
    elif model_type == 'fusion':
        logger.info("Starting fusion model training...")
        logger.info("Using pre-trained face and voice encoders if available")
        trainer, history = train_fusion_model(resume=resume)
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
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Run data analysis')
    
    args = parser.parse_args()
    
    if args.command == 'server':
        run_server()
    elif args.command == 'train':
        train_model(args.model, resume=args.resume)
    elif args.command == 'analyze':
        run_analysis()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()


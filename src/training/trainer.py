"""
Training Module for Age Prediction Models
Includes trainers for face, voice, and fusion models
"""
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from datetime import datetime
from typing import Dict, Optional

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import DEVICE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, MODELS_DIR, LOGS_DIR


class Trainer:
    """Base trainer class"""
    
    def __init__(self, model, train_loader, val_loader, criterion,
                 learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                 epochs=EPOCHS, device=DEVICE, model_name='model',
                 use_amp=True):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.device = device
        self.epochs = epochs
        self.model_name = model_name
        self.use_amp = use_amp and device.type == 'cuda'
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=learning_rate * 0.01
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'train_acc': [],
            'val_acc': []
        }
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_mae = 0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        
        for batch in pbar:
            self.optimizer.zero_grad()
            
            # Forward pass (implemented in subclass)
            loss_dict, predictions, targets = self._forward_batch(batch, train=True)
            loss = loss_dict['total']
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # Metrics
            total_loss += loss.item() * len(targets['age'])
            total_mae += torch.abs(predictions['age'] - targets['age']).sum().item()
            
            if 'age_group' in predictions:
                preds = torch.argmax(predictions['age_group_logits'], dim=1)
                total_correct += (preds == targets['age_group']).sum().item()
            
            total_samples += len(targets['age'])
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'mae': f"{(total_mae/total_samples):.2f}"
            })
        
        avg_loss = total_loss / total_samples
        avg_mae = total_mae / total_samples
        avg_acc = total_correct / total_samples if total_correct > 0 else 0
        
        return avg_loss, avg_mae, avg_acc
    
    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_mae = 0
        total_correct = 0
        total_samples = 0
        
        for batch in tqdm(self.val_loader, desc='Validating', leave=False):
            loss_dict, predictions, targets = self._forward_batch(batch, train=False)
            
            total_loss += loss_dict['total'].item() * len(targets['age'])
            total_mae += torch.abs(predictions['age'] - targets['age']).sum().item()
            
            if 'age_group_logits' in predictions:
                preds = torch.argmax(predictions['age_group_logits'], dim=1)
                total_correct += (preds == targets['age_group']).sum().item()
            
            total_samples += len(targets['age'])
        
        avg_loss = total_loss / total_samples
        avg_mae = total_mae / total_samples
        avg_acc = total_correct / total_samples if total_correct > 0 else 0
        
        return avg_loss, avg_mae, avg_acc
    
    def _forward_batch(self, batch, train=True):
        """Forward pass - to be implemented by subclass"""
        raise NotImplementedError
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint_path = MODELS_DIR / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, filename):
        """Load model checkpoint and return the starting epoch"""
        checkpoint_path = MODELS_DIR / filename
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore training state
        self.best_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.history = checkpoint.get('history', self.history)
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Resuming from epoch {self.best_epoch}, best_val_loss: {self.best_val_loss:.4f}")
        
        return self.best_epoch
    
    def train(self, save_best=True, early_stopping=10, start_epoch=0):
        """Full training loop with resume support"""
        logger.info(f"Starting training for {self.epochs} epochs (starting from epoch {start_epoch + 1})")
        logger.info(f"Device: {self.device}, AMP: {self.use_amp}")
        
        patience_counter = 0
        
        for epoch in range(start_epoch, self.epochs):
            logger.info(f"\nEpoch {epoch+1}/{self.epochs}")
            
            # Train
            train_loss, train_mae, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_mae, val_acc = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mae'].append(train_mae)
            self.history['val_mae'].append(val_mae)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            logger.info(f"Train - Loss: {train_loss:.4f}, MAE: {train_mae:.2f}, Acc: {train_acc:.4f}")
            logger.info(f"Val   - Loss: {val_loss:.4f}, MAE: {val_mae:.2f}, Acc: {val_acc:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                patience_counter = 0
                
                if save_best:
                    self.save_checkpoint(f"{self.model_name}_best.pth")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        logger.info(f"\nTraining complete. Best model at epoch {self.best_epoch}")
        return self.history


class FaceTrainer(Trainer):
    """Trainer for face age prediction model"""
    
    def _forward_batch(self, batch, train=True):
        images = batch['image'].to(self.device)
        targets = {
            'age': batch['age'].to(self.device),
            'age_group': batch['age_group'].to(self.device),
            'is_adult': batch['is_adult'].to(self.device)
        }
        
        if train and self.use_amp:
            with autocast():
                predictions = self.model(images)
                loss_dict = self.criterion(predictions, targets)
        else:
            predictions = self.model(images)
            loss_dict = self.criterion(predictions, targets)
        
        return loss_dict, predictions, targets


class VoiceTrainer(Trainer):
    """Trainer for voice age prediction model"""
    
    def _forward_batch(self, batch, train=True):
        features = batch['features'].to(self.device)
        targets = {
            'age': batch['age'].to(self.device),
            'age_group': batch['age_group'].to(self.device),
            'is_adult': batch['is_adult'].to(self.device)
        }
        
        if train and self.use_amp:
            with autocast():
                predictions = self.model(features)
                loss_dict = self.criterion(predictions, targets)
        else:
            predictions = self.model(features)
            loss_dict = self.criterion(predictions, targets)
        
        return loss_dict, predictions, targets


def train_face_model(resume=False):
    """Train face age prediction model
    
    Args:
        resume: If True, resume training from the last saved checkpoint
    """
    from src.data.data_analysis import DataAnalyzer
    from src.data.face_dataset import get_face_dataloaders
    from src.models.face_model import create_face_model, FaceAgeLoss
    
    # Prepare data
    analyzer = DataAnalyzer()
    face_data = analyzer.prepare_face_data()
    loaders = get_face_dataloaders(face_data, batch_size=16, num_workers=0)  # 0 for Windows compatibility
    
    # Create model and loss
    model = create_face_model(backbone='efficientnet_b0', pretrained=True)
    criterion = FaceAgeLoss()
    
    # Create trainer
    trainer = FaceTrainer(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        criterion=criterion,
        epochs=12,
        model_name='face_age_model',
        use_amp=False
    )
    
    # Resume from checkpoint if requested
    start_epoch = 0
    if resume:
        checkpoint_path = MODELS_DIR / 'face_age_model_best.pth'
        if checkpoint_path.exists():
            start_epoch = trainer.load_checkpoint('face_age_model_best.pth')
            logger.info(f"Resuming face model training from epoch {start_epoch + 1}")
        else:
            logger.warning("No checkpoint found, starting from scratch")
    
    # Train
    history = trainer.train(save_best=True, early_stopping=5, start_epoch=start_epoch)
    
    return trainer, history


def train_voice_model(resume=False):
    """Train voice age prediction model
    
    Args:
        resume: If True, resume training from the last saved checkpoint
    """
    from src.data.data_analysis import DataAnalyzer
    from src.data.audio_dataset import get_audio_dataloaders
    from src.models.voice_model import create_voice_model, VoiceAgeLoss
    
    # Prepare data
    analyzer = DataAnalyzer()
    audio_data = analyzer.prepare_audio_data()
    loaders = get_audio_dataloaders(audio_data, batch_size=16, num_workers=0)  # 0 for Windows compatibility
    
    # Create model and loss
    model = create_voice_model()
    criterion = VoiceAgeLoss()
    
    # Create trainer
    trainer = VoiceTrainer(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        criterion=criterion,
        epochs=12,
        model_name='voice_age_model',
        use_amp=False
    )
    
    # Resume from checkpoint if requested
    start_epoch = 0
    if resume:
        checkpoint_path = MODELS_DIR / 'voice_age_model_best.pth'
        if checkpoint_path.exists():
            start_epoch = trainer.load_checkpoint('voice_age_model_best.pth')
            logger.info(f"Resuming voice model training from epoch {start_epoch + 1}")
        else:
            logger.warning("No checkpoint found, starting from scratch")

    # Train
    history = trainer.train(save_best=True, early_stopping=5, start_epoch=start_epoch)
    
    return trainer, history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["face", "voice"], default="face")
    args = parser.parse_args()
    if args.model == "face":
        train_face_model()
    elif args.model == "voice":
        train_voice_model()



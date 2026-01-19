"""
Face Dataset for UTKFace
PyTorch Dataset implementation for face age prediction
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from loguru import logger

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import IMAGE_DATA_DIR, FACE_IMAGE_SIZE, BATCH_SIZE, AGE_GROUPS


class UTKFaceDataset(Dataset):
    """PyTorch Dataset for UTKFace images"""
    
    def __init__(self, dataframe, transform=None, augment=False):
        """
        Args:
            dataframe: DataFrame with 'filepath', 'age', 'age_group' columns
            transform: Optional torchvision transforms
            augment: Whether to apply data augmentation
        """
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform
        self.augment = augment
        
        # Age group encoding
        self.age_groups = ['child', 'teen', 'adult', 'senior']
        self.age_group_to_idx = {g: i for i, g in enumerate(self.age_groups)}
        
        # Build transforms
        if self.transform is None:
            if augment:
                self.transform = self._get_augment_transform()
            else:
                self.transform = self._get_base_transform()
    
    def _get_base_transform(self):
        """Basic transform for validation/test"""
        return transforms.Compose([
            transforms.Resize(FACE_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _get_augment_transform(self):
        """Augmented transform for training"""
        return transforms.Compose([
            transforms.Resize((int(FACE_IMAGE_SIZE[0] * 1.1), int(FACE_IMAGE_SIZE[1] * 1.1))),
            transforms.RandomCrop(FACE_IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load image
        img_path = row['filepath']
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Error loading image {img_path}: {e}")
            # Return a blank image on error
            image = Image.new('RGB', FACE_IMAGE_SIZE, color='gray')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get labels
        age = torch.tensor(row['age'], dtype=torch.float32)
        age_group_idx = self.age_group_to_idx.get(row['age_group'], 2)  # default to adult
        age_group = torch.tensor(age_group_idx, dtype=torch.long)
        is_adult = torch.tensor(1 if row['is_adult'] else 0, dtype=torch.float32)
        
        return {
            'image': image,
            'age': age,
            'age_group': age_group,
            'is_adult': is_adult,
            'filepath': img_path
        }


def get_face_dataloaders(face_data_dict, batch_size=BATCH_SIZE, num_workers=4):
    """
    Create DataLoaders for train, val, test splits
    
    Args:
        face_data_dict: Dictionary with 'train', 'val', 'test' DataFrames
        batch_size: Batch size for loading
        num_workers: Number of data loading workers
    
    Returns:
        Dictionary with train, val, test DataLoaders
    """
    train_dataset = UTKFaceDataset(face_data_dict['train'], augment=True)
    val_dataset = UTKFaceDataset(face_data_dict['val'], augment=False)
    test_dataset = UTKFaceDataset(face_data_dict['test'], augment=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created DataLoaders - Train: {len(train_loader)} batches, "
                f"Val: {len(val_loader)} batches, Test: {len(test_loader)} batches")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'class_weights': face_data_dict.get('class_weights', None)
    }


class LiveFaceDataset(Dataset):
    """Dataset for processing live video frames"""
    
    def __init__(self, frames):
        """
        Args:
            frames: List of numpy arrays (BGR frames from OpenCV)
        """
        self.frames = frames
        self.transform = transforms.Compose([
            transforms.Resize(FACE_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        
        # Convert BGR to RGB
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = frame[:, :, ::-1]
        
        # Convert to PIL Image
        image = Image.fromarray(frame.astype('uint8'))
        
        # Apply transforms
        image = self.transform(image)
        
        return image


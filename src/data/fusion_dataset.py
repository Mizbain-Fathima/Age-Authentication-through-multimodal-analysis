"""
Fusion Dataset for Multimodal Age Prediction
Combines face images and audio features for joint training

Since UTKFace and Common Voice datasets don't have paired data from the same person,
we use age-matched synthetic pairing: pair face images with audio samples of similar age.
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from loguru import logger
import random

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import BATCH_SIZE, AGE_GROUPS
from src.data.face_dataset import UTKFaceDataset
from src.data.audio_dataset import CommonVoiceDataset, AudioFeatureExtractor
from src.data.data_analysis import DataAnalyzer


class FusionDataset(Dataset):
    """
    Dataset that provides paired face images and audio features.
    
    Pairing Strategy:
        Since we don't have true paired data, we use age-matched pairing:
        - For each sample, randomly pair a face image with an audio sample
          of the same age group (or closest age within tolerance)
        - This allows the model to learn multimodal features even without
          true identity-matched pairs
    """
    
    def __init__(self, face_df, audio_df, augment=False, age_tolerance=5):
        """
        Args:
            face_df: DataFrame with face image metadata
            audio_df: DataFrame with audio metadata
            augment: Whether to apply data augmentation
            age_tolerance: Age difference tolerance for pairing (years)
        """
        self.face_df = face_df.reset_index(drop=True)
        self.audio_df = audio_df.reset_index(drop=True)
        self.augment = augment
        self.age_tolerance = age_tolerance
        
        # Create face dataset for image loading/transforms
        self.face_dataset = UTKFaceDataset(face_df, augment=augment)
        
        # Create audio dataset for feature extraction
        self.audio_dataset = CommonVoiceDataset(audio_df, augment=augment)
        
        # Build age-based index for efficient pairing
        self._build_age_index()
        
        logger.info(f"Created FusionDataset with {len(self.face_df)} face samples "
                   f"and {len(self.audio_df)} audio samples")
    
    def _build_age_index(self):
        """Build index of samples by age for efficient pairing"""
        # Index audio samples by age
        self.audio_by_age = {}
        for idx, row in self.audio_df.iterrows():
            age = int(row['numeric_age'])
            if age not in self.audio_by_age:
                self.audio_by_age[age] = []
            self.audio_by_age[age].append(idx)
        
        # Index face samples by age group
        self.face_by_age_group = {}
        for idx, row in self.face_df.iterrows():
            age_group = row['age_group']
            if age_group not in self.face_by_age_group:
                self.face_by_age_group[age_group] = []
            self.face_by_age_group[age_group].append(idx)
    
    def _find_audio_match(self, face_age, face_age_group):
        """Find a matching audio sample based on age"""
        face_age = int(face_age)
        
        # Try exact age match first
        if face_age in self.audio_by_age and self.audio_by_age[face_age]:
            return random.choice(self.audio_by_age[face_age])
        
        # Try within tolerance
        for offset in range(1, self.age_tolerance + 1):
            for age in [face_age + offset, face_age - offset]:
                if age in self.audio_by_age and self.audio_by_age[age]:
                    return random.choice(self.audio_by_age[age])
        
        # Fallback: pick any audio from same age group
        audio_group_samples = self.audio_df[self.audio_df['age_group'] == face_age_group]
        if len(audio_group_samples) > 0:
            return audio_group_samples.sample(1).index[0]
        
        # Last resort: random audio sample
        return random.randint(0, len(self.audio_df) - 1)
    
    def __len__(self):
        return len(self.face_df)
    
    def __getitem__(self, idx):
        # Get face sample
        face_sample = self.face_dataset[idx]
        face_row = self.face_df.iloc[idx]
        
        # Find matching audio sample by age
        audio_idx = self._find_audio_match(face_row['age'], face_row['age_group'])
        audio_sample = self.audio_dataset[audio_idx]
        
        # Use face sample's labels as ground truth (since face age is more reliable)
        return {
            'face': face_sample['image'],
            'voice': audio_sample['features'],
            'age': face_sample['age'],
            'age_group': face_sample['age_group'],
            'is_adult': face_sample['is_adult'],
            'face_filepath': face_sample['filepath'],
            'audio_filepath': audio_sample['filepath']
        }


def fusion_collate_fn(batch):
    """
    Custom collate function for fusion dataset.
    Defined at module level for multiprocessing compatibility.
    """
    faces = torch.stack([item['face'] for item in batch])
    voices = torch.stack([item['voice'] for item in batch])
    ages = torch.stack([item['age'] for item in batch])
    age_groups = torch.stack([item['age_group'] for item in batch])
    is_adults = torch.stack([item['is_adult'] for item in batch])
    face_paths = [item['face_filepath'] for item in batch]
    audio_paths = [item['audio_filepath'] for item in batch]
    
    return {
        'face': faces,
        'voice': voices,
        'age': ages,
        'age_group': age_groups,
        'is_adult': is_adults,
        'face_filepath': face_paths,
        'audio_filepath': audio_paths
    }


def get_fusion_dataloaders(batch_size=BATCH_SIZE, num_workers=0, age_tolerance=5):
    """
    Create DataLoaders for fusion model training.
    
    Args:
        batch_size: Batch size for training
        num_workers: Number of data loading workers (0 for Windows)
        age_tolerance: Age difference tolerance for pairing
    
    Returns:
        Dictionary with train, val, test DataLoaders and metadata
    """
    # Prepare both datasets
    analyzer = DataAnalyzer()
    face_data = analyzer.prepare_face_data()
    audio_data = analyzer.prepare_audio_data()
    
    # Create fusion datasets
    train_dataset = FusionDataset(
        face_data['train'],
        audio_data['train'],
        augment=True,
        age_tolerance=age_tolerance
    )
    
    val_dataset = FusionDataset(
        face_data['val'],
        audio_data['val'],
        augment=False,
        age_tolerance=age_tolerance
    )
    
    test_dataset = FusionDataset(
        face_data['test'],
        audio_data['test'],
        augment=False,
        age_tolerance=age_tolerance
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=fusion_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=fusion_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=fusion_collate_fn
    )
    
    logger.info(f"Created Fusion DataLoaders - Train: {len(train_loader)} batches, "
                f"Val: {len(val_loader)} batches, Test: {len(test_loader)} batches")
    
    # Combine class weights from both modalities
    combined_weights = {}
    for key in face_data.get('class_weights', {}).keys():
        face_w = face_data.get('class_weights', {}).get(key, 1.0)
        audio_w = audio_data.get('class_weights', {}).get(key, 1.0)
        combined_weights[key] = (face_w + audio_w) / 2
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'class_weights': combined_weights
    }


class LiveFusionProcessor:
    """Process live face + audio input for inference"""
    
    def __init__(self):
        from src.data.face_dataset import UTKFaceDataset
        self.audio_extractor = AudioFeatureExtractor()
        # For face preprocessing, we'll use torchvision transforms
        import torchvision.transforms as T
        from src.config import FACE_IMAGE_SIZE
        
        self.face_transform = T.Compose([
            T.ToPILImage(),
            T.Resize(FACE_IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def process_face_frame(self, frame):
        """
        Process a single face frame for inference
        
        Args:
            frame: BGR image (numpy array) or RGB image
        
        Returns:
            Tensor ready for model input (1, 3, H, W)
        """
        import cv2
        
        # Convert BGR to RGB if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
        
        # Apply transforms
        face_tensor = self.face_transform(frame_rgb)
        return face_tensor.unsqueeze(0)  # Add batch dimension
    
    def process_audio(self, audio_data, sample_rate=16000):
        """
        Process audio data for inference
        
        Args:
            audio_data: Audio waveform as numpy array
            sample_rate: Sample rate of the audio
        
        Returns:
            Tensor ready for model input (1, 2, H, W)
        """
        import librosa
        from src.config import SAMPLE_RATE, N_MELS
        
        # Resample if needed
        if sample_rate != SAMPLE_RATE:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
        
        # Extract features
        mfcc = self.audio_extractor.extract_mfcc(audio_data)
        mel_spec = self.audio_extractor.extract_mel_spectrogram(audio_data)
        
        # Pad MFCC to match mel spectrogram
        mfcc_padded = np.zeros((N_MELS, mfcc.shape[1]), dtype=mfcc.dtype)
        mfcc_padded[:mfcc.shape[0], :] = mfcc
        
        # Stack features
        combined = np.stack([mfcc_padded, mel_spec], axis=0)
        features = torch.tensor(combined, dtype=torch.float32).unsqueeze(0)
        
        return features
    
    def process_paired_input(self, face_frame, audio_data, sample_rate=16000):
        """
        Process both face and audio for fusion model inference
        
        Args:
            face_frame: Face image (numpy array)
            audio_data: Audio waveform (numpy array)
            sample_rate: Audio sample rate
        
        Returns:
            Dictionary with 'face' and 'voice' tensors
        """
        face_tensor = self.process_face_frame(face_frame)
        audio_tensor = self.process_audio(audio_data, sample_rate)
        
        return {
            'face': face_tensor,
            'voice': audio_tensor
        }


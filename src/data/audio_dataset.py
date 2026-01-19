"""
Audio Dataset for Mozilla Common Voice
PyTorch Dataset implementation for voice age prediction
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import (
    AUDIO_DATA_DIR, SAMPLE_RATE, N_MFCC, N_MELS, 
    HOP_LENGTH, N_FFT, MAX_AUDIO_LENGTH, BATCH_SIZE, AGE_GROUPS
)


class AudioFeatureExtractor:
    """Extract audio features: MFCC and Log-Mel Spectrograms"""
    
    def __init__(self, sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC, n_mels=N_MELS,
                 hop_length=HOP_LENGTH, n_fft=N_FFT, max_length=MAX_AUDIO_LENGTH):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.max_length = max_length
        self.max_frames = int(max_length * sample_rate / hop_length)
    
    def load_audio(self, audio_path):
        """Load and preprocess audio file"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            # Trim silence
            y, _ = librosa.effects.trim(y, top_db=20)
            
            # Pad or truncate to max length
            max_samples = int(self.max_length * self.sample_rate)
            if len(y) > max_samples:
                y = y[:max_samples]
            elif len(y) < max_samples:
                y = np.pad(y, (0, max_samples - len(y)), mode='constant')
            
            return y
        except Exception as e:
            logger.warning(f"Error loading audio {audio_path}: {e}")
            return np.zeros(int(self.max_length * self.sample_rate))
    
    def extract_mfcc(self, y):
        """Extract MFCC features"""
        mfcc = librosa.feature.mfcc(
            y=y, 
            sr=self.sample_rate, 
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Add deltas
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Stack features
        features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
        
        # Normalize
        features = (features - features.mean(axis=1, keepdims=True)) / (features.std(axis=1, keepdims=True) + 1e-8)
        
        # Pad/truncate to fixed size
        if features.shape[1] > self.max_frames:
            features = features[:, :self.max_frames]
        elif features.shape[1] < self.max_frames:
            features = np.pad(features, ((0, 0), (0, self.max_frames - features.shape[1])), mode='constant')
        
        return features
    
    def extract_mel_spectrogram(self, y):
        """Extract Log-Mel Spectrogram"""
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-8)
        
        # Pad/truncate to fixed size
        if log_mel_spec.shape[1] > self.max_frames:
            log_mel_spec = log_mel_spec[:, :self.max_frames]
        elif log_mel_spec.shape[1] < self.max_frames:
            log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, self.max_frames - log_mel_spec.shape[1])), mode='constant')
        
        return log_mel_spec
    
    def extract_all_features(self, audio_path):
        """Extract both MFCC and Mel Spectrogram features"""
        y = self.load_audio(audio_path)
        mfcc = self.extract_mfcc(y)
        mel_spec = self.extract_mel_spectrogram(y)
        
        return {
            'mfcc': mfcc,  # Shape: (n_mfcc * 3, max_frames)
            'mel_spec': mel_spec  # Shape: (n_mels, max_frames)
        }


class CommonVoiceDataset(Dataset):
    """PyTorch Dataset for Common Voice audio"""
    
    def __init__(self, dataframe, feature_type='combined', augment=False):
        """
        Args:
            dataframe: DataFrame with audio metadata
            feature_type: 'mfcc', 'mel', or 'combined'
            augment: Whether to apply data augmentation
        """
        self.data = dataframe.reset_index(drop=True)
        self.feature_type = feature_type
        self.augment = augment
        self.feature_extractor = AudioFeatureExtractor()
        
        # Age group encoding
        self.age_groups = ['child', 'teen', 'adult', 'senior']
        self.age_group_to_idx = {g: i for i, g in enumerate(self.age_groups)}

        self.cache_dir = Path("cache/audio_features")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    
    def _augment_audio(self, y):
        """Apply audio augmentation"""
        if not self.augment:
            return y
        
        # Random time stretch
        if np.random.random() < 0.3:
            rate = np.random.uniform(0.9, 1.1)
            y = librosa.effects.time_stretch(y, rate=rate)
        
        # Random pitch shift
        if np.random.random() < 0.3:
            n_steps = np.random.uniform(-2, 2)
            y = librosa.effects.pitch_shift(y, sr=SAMPLE_RATE, n_steps=n_steps)
        
        # Add noise
        if np.random.random() < 0.2:
            noise = np.random.randn(len(y)) * 0.005
            y = y + noise
        
        # Random gain
        if np.random.random() < 0.3:
            gain = np.random.uniform(0.8, 1.2)
            y = y * gain
        
        return y
    
    def __len__(self):
        return len(self.data)
    
    def _resolve_audio_path(self, filename):
        """Resolve audio path, handling nested directory structure"""
        # Try direct path first
        direct_path = AUDIO_DATA_DIR / filename
        if direct_path.exists():
            return direct_path
        # Handle nested directory structure (e.g., cv-other-train/cv-other-train/sample.mp3)
        parts = filename.replace('\\', '/').split('/')
        if len(parts) >= 2:
            nested_path = AUDIO_DATA_DIR / parts[0] / filename
            if nested_path.exists():
                return nested_path
        return direct_path  # Return original path for error handling
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        audio_path = self._resolve_audio_path(row["filename"])

        cache_file = self.cache_dir / f"{audio_path.stem}.npz"

        # -------- CACHE LOAD --------
        if cache_file.exists() and not self.augment:
            cached = np.load(cache_file)
            mfcc = cached["mfcc"]
            mel = cached["mel"]
        else:
            y = self.feature_extractor.load_audio(str(audio_path))

            if self.augment:
                y = self._augment_audio(y)

            mfcc = self.feature_extractor.extract_mfcc(y)
            mel = self.feature_extractor.extract_mel_spectrogram(y)

            # Save cache ONLY for non-augmented samples
            if not self.augment:
                np.savez_compressed(cache_file, mfcc=mfcc, mel=mel)

        # -------- FEATURE FORMAT --------
        if self.feature_type == "mfcc":
            features = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)
        elif self.feature_type == "mel":
            features = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        else:  # combined
            # Pad MFCC to match mel spectrogram dimensions (N_MFCC=40 -> N_MELS=128)
            mfcc_padded = np.zeros((N_MELS, mfcc.shape[1]), dtype=mfcc.dtype)
            mfcc_padded[:mfcc.shape[0], :] = mfcc
            combined = np.stack([mfcc_padded, mel], axis=0)
            features = torch.tensor(combined, dtype=torch.float32)

        return {
            "features": features,
            "age": torch.tensor(row["numeric_age"], dtype=torch.float32),
            "age_group": torch.tensor(
                self.age_group_to_idx.get(row["age_group"], 2), dtype=torch.long
            ),
            "is_adult": torch.tensor(int(row["is_adult"]), dtype=torch.float32),
            "text": row["text"],
            "filepath": str(audio_path)
        }


def audio_collate_fn(batch):
    """
    Custom collate function to handle variable length texts.
    Defined at module level to allow pickling for multiprocessing on Windows.
    """
    features = torch.stack([item['features'] for item in batch])
    ages = torch.stack([item['age'] for item in batch])
    age_groups = torch.stack([item['age_group'] for item in batch])
    is_adults = torch.stack([item['is_adult'] for item in batch])
    texts = [item['text'] for item in batch]
    filepaths = [item['filepath'] for item in batch]
    
    return {
        'features': features,
        'age': ages,
        'age_group': age_groups,
        'is_adult': is_adults,
        'text': texts,
        'filepath': filepaths
    }


def get_audio_dataloaders(audio_data_dict, batch_size=BATCH_SIZE, num_workers=0, feature_type='combined'):
    """
    Create DataLoaders for train, val, test splits
    
    Args:
        audio_data_dict: Dictionary with 'train', 'val', 'test' DataFrames
        batch_size: Batch size for loading
        num_workers: Number of data loading workers (0 for Windows compatibility)
        feature_type: Type of audio features
    
    Returns:
        Dictionary with train, val, test DataLoaders
    """
    train_dataset = CommonVoiceDataset(audio_data_dict['train'], feature_type=feature_type, augment=True)
    val_dataset = CommonVoiceDataset(audio_data_dict['val'], feature_type=feature_type, augment=False)
    test_dataset = CommonVoiceDataset(audio_data_dict['test'], feature_type=feature_type, augment=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=audio_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=audio_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=audio_collate_fn
    )
    
    logger.info(f"Created Audio DataLoaders - Train: {len(train_loader)} batches, "
                f"Val: {len(val_loader)} batches, Test: {len(test_loader)} batches")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'class_weights': audio_data_dict.get('class_weights', None)
    }


class LiveAudioProcessor:
    """Process live audio input for inference"""
    
    def __init__(self):
        self.feature_extractor = AudioFeatureExtractor()
    
    def process_audio_bytes(self, audio_bytes, sample_rate=SAMPLE_RATE):
        """
        Process raw audio bytes for inference
        
        Args:
            audio_bytes: Raw audio data as bytes or numpy array
            sample_rate: Sample rate of the audio
        
        Returns:
            Tensor ready for model input
        """
        import io
        import soundfile as sf
        
        if isinstance(audio_bytes, bytes):
            # Load from bytes
            audio_io = io.BytesIO(audio_bytes)
            y, sr = sf.read(audio_io)
            
            # Resample if needed
            if sr != SAMPLE_RATE:
                y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        else:
            y = audio_bytes
        
        # Ensure mono
        if len(y.shape) > 1:
            y = y.mean(axis=1)
        
        # Extract features
        mfcc = self.feature_extractor.extract_mfcc(y)
        mel_spec = self.feature_extractor.extract_mel_spectrogram(y)
        
        # Combine features
        combined = np.stack([
            mfcc[:N_MELS, :],
            mel_spec
        ], axis=0)
        
        features = torch.tensor(combined, dtype=torch.float32).unsqueeze(0)  # Add batch dim
        
        return features


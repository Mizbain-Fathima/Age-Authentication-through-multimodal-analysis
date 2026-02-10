"""
Multimodal dataset: pairs face and audio samples by age group for joint training.
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import (
    FACE_IMAGE_SIZE,
    RANDOM_SEED,
    AUDIO_DATA_DIR,
    KIDS_AUDIO_DATA_DIR,
    N_MELS,
)
from src.data.audio_dataset import AudioFeatureExtractor


class MultimodalDataset(Dataset):
    """
    Pairs face and audio samples by age group. Uses face labels (age, age_group, is_adult).
    Pairing is deterministic (fixed seed).
    """

    AGE_GROUPS = ["child", "teen", "adult", "senior"]

    def __init__(self, face_df, audio_df, augment=False, seed=RANDOM_SEED):
        self.face_df = face_df.reset_index(drop=True)
        self.audio_df = audio_df.reset_index(drop=True)
        self.augment = augment
        self.seed = seed
        self.age_group_to_idx = {g: i for i, g in enumerate(self.AGE_GROUPS)}

        self._build_audio_index()
        self._build_pairs()

        self.face_transform = self._get_face_transform()
        self.audio_extractor = AudioFeatureExtractor()
        self.audio_dim = 2 * N_MELS * self.audio_extractor.max_frames

    def _build_audio_index(self):
        self.audio_by_group = {}
        for i in range(len(self.audio_df)):
            row = self.audio_df.iloc[i]
            g = row["age_group"]
            if g not in self.audio_by_group:
                self.audio_by_group[g] = []
            self.audio_by_group[g].append(i)

    def _build_pairs(self):
        rng = random.Random(self.seed)
        self.pairs = []
        for face_idx in range(len(self.face_df)):
            row = self.face_df.iloc[face_idx]
            g = row["age_group"]
            if g in self.audio_by_group and self.audio_by_group[g]:
                audio_idx = rng.choice(self.audio_by_group[g])
            else:
                audio_idx = rng.randint(0, len(self.audio_df) - 1) if len(self.audio_df) > 0 else 0
            self.pairs.append((face_idx, audio_idx))

    def _get_face_transform(self):
        if self.augment:
            return transforms.Compose([
                transforms.Resize((int(FACE_IMAGE_SIZE[0] * 1.1), int(FACE_IMAGE_SIZE[1] * 1.1))),
                transforms.RandomCrop(FACE_IMAGE_SIZE),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        return transforms.Compose([
            transforms.Resize(FACE_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _resolve_audio_path(self, row):
        if row.get("source") == "kids_audio" and row.get("filepath") and Path(row["filepath"]).exists():
            return Path(row["filepath"])
        fn = row.get("filename", "")
        direct = AUDIO_DATA_DIR / fn
        if direct.exists():
            return direct
        parts = fn.replace("\\", "/").split("/")
        if len(parts) >= 2:
            nested = AUDIO_DATA_DIR / parts[0] / fn
            if nested.exists():
                return nested
        return AUDIO_DATA_DIR / fn

    def _get_audio_embedding(self, audio_idx):
        row = self.audio_df.iloc[audio_idx]
        path = self._resolve_audio_path(row)
        try:
            y = self.audio_extractor.load_audio(str(path))
            mfcc = self.audio_extractor.extract_mfcc(y)
            mel = self.audio_extractor.extract_mel_spectrogram(y)
        except Exception:
            n_mfcc3 = self.audio_extractor.n_mfcc * 3
            mfcc = np.zeros((n_mfcc3, self.audio_extractor.max_frames), dtype=np.float32)
            mel = np.zeros((self.audio_extractor.n_mels, self.audio_extractor.max_frames), dtype=np.float32)
        mfcc_pad = np.zeros((N_MELS, mfcc.shape[1]), dtype=mfcc.dtype)
        mfcc_pad[: mfcc.shape[0], :] = mfcc
        combined = np.stack([mfcc_pad, mel], axis=0)
        return torch.tensor(combined.flatten(), dtype=torch.float32)

    def __len__(self):
        return len(self.face_df)

    def __getitem__(self, idx):
        face_idx, audio_idx = self.pairs[idx]
        face_row = self.face_df.iloc[face_idx]

        img_path = face_row["filepath"]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", FACE_IMAGE_SIZE, color="gray")
        image = self.face_transform(image)

        audio_embedding = self._get_audio_embedding(audio_idx)

        age = float(face_row["age"])
        age_group = self.age_group_to_idx.get(face_row["age_group"], 2)
        is_adult = 1.0 if face_row["is_adult"] else 0.0

        return {
            "image": image,
            "audio": audio_embedding,
            "age": torch.tensor(age, dtype=torch.float32),
            "age_group": torch.tensor(age_group, dtype=torch.long),
            "is_adult": torch.tensor(is_adult, dtype=torch.float32),
        }


def get_multimodal_dataloaders(face_data, audio_data, batch_size=32, num_workers=0, seed=RANDOM_SEED):
    """Build train/val/test DataLoaders for MultimodalDataset."""
    from torch.utils.data import DataLoader

    train_ds = MultimodalDataset(
        face_data["train"], audio_data["train"], augment=True, seed=seed
    )
    val_ds = MultimodalDataset(
        face_data["val"], audio_data["val"], augment=False, seed=seed
    )
    test_ds = MultimodalDataset(
        face_data["test"], audio_data["test"], augment=False, seed=seed
    )

    def collate(batch):
        return {
            "image": torch.stack([b["image"] for b in batch]),
            "audio": torch.stack([b["audio"] for b in batch]),
            "age": torch.stack([b["age"] for b in batch]),
            "age_group": torch.stack([b["age_group"] for b in batch]),
            "is_adult": torch.stack([b["is_adult"] for b in batch]),
        }

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        drop_last=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate
    )
    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "audio_dim": train_ds.audio_dim,
    }

"""
Single joint end-to-end multimodal model: face + audio -> shared representation -> prediction heads.
"""
import torch
import torch.nn as nn
from torchvision import models

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import DEVICE

FACE_DIM = 256
AUDIO_DIM = 256
FUSION_HIDDEN = 256
DROPOUT = 0.3


class MultimodalAgeModel(nn.Module):
    """
    Joint multimodal model: face encoder (EfficientNet-B0 -> 256) + audio encoder (MLP -> 256),
    concat -> shared representation -> single set of prediction heads.
    """

    def __init__(self, audio_input_dim, face_backbone="efficientnet_b0", dropout=DROPOUT):
        super().__init__()
        if face_backbone == "efficientnet_b0":
            backbone = models.efficientnet_b0(weights="IMAGENET1K_V1")
            face_feat_dim = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {face_backbone}")
        self.face_encoder = backbone
        self.face_proj = nn.Sequential(
            nn.Linear(face_feat_dim, FACE_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, AUDIO_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.fusion = nn.Sequential(
            nn.Linear(FACE_DIM + AUDIO_DIM, FUSION_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.age_head = nn.Linear(FUSION_HIDDEN, 1)
        self.age_group_head = nn.Linear(FUSION_HIDDEN, 4)
        self.adult_head = nn.Linear(FUSION_HIDDEN, 1)

    def forward(self, image, audio):
        face_feat = self.face_encoder(image)
        face_feat = self.face_proj(face_feat)

        audio_feat = self.audio_encoder(audio)

        joint = torch.cat([face_feat, audio_feat], dim=-1)
        shared = self.fusion(joint)

        age_pred = self.age_head(shared).squeeze(-1).clamp(0, 100)
        age_group_logits = self.age_group_head(shared)
        adult_logit = self.adult_head(shared).squeeze(-1)

        return {
            "age": age_pred,
            "age_group_logits": age_group_logits,
            "is_adult_logit": adult_logit,
        }


def create_multimodal_model(audio_input_dim, device=DEVICE):
    model = MultimodalAgeModel(audio_input_dim=audio_input_dim)
    return model.to(device)


def load_multimodal_model(checkpoint_path, device=DEVICE):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict") or ckpt
    # Treat as fusion if model_type says so, or if state dict has fusion keys (e.g. voice_encoder)
    is_fusion = (
        ckpt.get("model_type") == "fusion"
        or (isinstance(state, dict) and any(k.startswith("voice_encoder") for k in state))
    )
    if is_fusion:
        from src.models.fusion_model import load_fusion_model
        return load_fusion_model(checkpoint_path, device)
    audio_dim = ckpt.get("audio_dim")
    if audio_dim is None:
        from src.config import N_MELS, SAMPLE_RATE, HOP_LENGTH, MAX_AUDIO_LENGTH
        max_frames = int(MAX_AUDIO_LENGTH * SAMPLE_RATE / HOP_LENGTH)
        audio_dim = 2 * N_MELS * max_frames
    model = MultimodalAgeModel(audio_input_dim=audio_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model

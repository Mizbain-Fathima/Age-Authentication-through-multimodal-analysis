"""
Late Fusion Baseline — STANDARD multimodal baseline for comparison.

This model is a BASELINE used for comparison against a proposed
joint end-to-end multimodal age authentication model.

- Uses pretrained face encoder + audio encoder (existing architectures).
- Fusion: concatenation or weighted average of embeddings (no cross-attention).
- Shallow MLP classifier/regressor on top.
- Encoders are frozen; no joint backpropagation across encoders.
- Do NOT use: cross-attention, joint end-to-end multimodal learning.
"""
import torch
import torch.nn as nn
from loguru import logger

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import FUSION_HIDDEN_DIM, FUSION_DROPOUT, DEVICE
from src.models.face_model import FaceAgeModel
from src.models.voice_model import VoiceAgeModel


class LateFusionBaseline(nn.Module):
    """
    Late Fusion Baseline: pretrained face + audio encoders, then concatenation
    or weighted average of embeddings, then a shallow MLP for age prediction.
    No cross-attention; encoders are frozen.
    """

    def __init__(
        self,
        face_encoder: nn.Module,
        voice_encoder: nn.Module,
        fusion_method: str = "concat",
        hidden_dim: int = FUSION_HIDDEN_DIM,
        dropout: float = FUSION_DROPOUT,
        freeze_encoders: bool = True,
    ):
        super().__init__()
        self.fusion_method = fusion_method
        self.face_encoder = face_encoder
        self.voice_encoder = voice_encoder

        face_dim = self.face_encoder.get_feature_dim()
        voice_dim = self.voice_encoder.get_feature_dim()

        if freeze_encoders:
            for p in self.face_encoder.parameters():
                p.requires_grad = False
            for p in self.voice_encoder.parameters():
                p.requires_grad = False
            logger.info("Late Fusion Baseline: froze face and voice encoders")

        if fusion_method == "concat":
            fusion_input_dim = face_dim + voice_dim
        elif fusion_method == "weighted_avg":
            fusion_input_dim = face_dim  # project both to same dim then average
            self.face_proj = nn.Linear(face_dim, hidden_dim)
            self.voice_proj = nn.Linear(voice_dim, hidden_dim)
            fusion_input_dim = hidden_dim
        else:
            raise ValueError(f"fusion_method must be 'concat' or 'weighted_avg', got {fusion_method}")

        self.fusion_input_dim = fusion_input_dim

        # Shallow MLP on top of fused features
        self.mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        mlp_out_dim = hidden_dim // 2

        self.age_head = nn.Linear(mlp_out_dim, 1)
        self.age_group_head = nn.Linear(mlp_out_dim, 4)
        self.adult_head = nn.Linear(mlp_out_dim, 1)

        logger.info(f"Late Fusion Baseline initialized (fusion={fusion_method}, hidden_dim={hidden_dim})")

    def forward(self, face_input, voice_input, return_details=False):
        # Extract features (encoders frozen)
        face_out = self.face_encoder(face_input, return_features=True)
        voice_out = self.voice_encoder(voice_input, return_features=True)
        face_feat = face_out["features"]
        voice_feat = voice_out["features"]

        if self.fusion_method == "concat":
            fused = torch.cat([face_feat, voice_feat], dim=-1)
        else:
            face_h = self.face_proj(face_feat)
            voice_h = self.voice_proj(voice_feat)
            fused = (face_h + voice_h) * 0.5

        h = self.mlp(fused)
        age = self.age_head(h).squeeze(-1).clamp(0, 100)
        age_group_logits = self.age_group_head(h)
        adult_logit = self.adult_head(h).squeeze(-1)

        out = {
            "age": age,
            "age_group_logits": age_group_logits,
            "is_adult_logit": adult_logit,
            "is_adult": torch.sigmoid(adult_logit),
        }
        if return_details:
            out["face_features"] = face_feat
            out["voice_features"] = voice_feat
        return out


def create_late_fusion_baseline(
    face_checkpoint=None,
    voice_checkpoint=None,
    fusion_method="concat",
    freeze_encoders=True,
    device=DEVICE,
):
    """Create Late Fusion Baseline; load encoders from checkpoints if provided."""
    from src.models.face_model import load_face_model
    from src.models.voice_model import load_voice_model

    if face_checkpoint and Path(face_checkpoint).exists():
        face_encoder = load_face_model(face_checkpoint, backbone="efficientnet_b0", device=device)
    else:
        face_encoder = FaceAgeModel(backbone="efficientnet_b0", pretrained=True)
    if voice_checkpoint and Path(voice_checkpoint).exists():
        voice_encoder = load_voice_model(voice_checkpoint, device=device)
    else:
        voice_encoder = VoiceAgeModel()

    model = LateFusionBaseline(
        face_encoder=face_encoder,
        voice_encoder=voice_encoder,
        fusion_method=fusion_method,
        freeze_encoders=freeze_encoders,
    )
    return model.to(device)

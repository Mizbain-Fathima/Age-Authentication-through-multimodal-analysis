"""
Multimodal fusion model: face encoder (FaceAgeModel backbone) + voice encoder (VoiceAgeModel backbone)
-> fusion block -> shared heads. Fully trainable end-to-end.
"""
import torch
import torch.nn as nn
from collections import OrderedDict

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import DEVICE, FACE_BACKBONE, FACE_DROPOUT, FUSION_HIDDEN_DIM, FUSION_DROPOUT, N_MELS, SAMPLE_RATE, HOP_LENGTH, MAX_AUDIO_LENGTH
from src.models.face_model import FaceAgeModel
from src.models.voice_model import VoiceAgeModel

FACE_FEAT_DIM = 256
VOICE_FEAT_DIM = 64


class MultimodalFusionModel(nn.Module):
    """
    End-to-end trainable fusion: face_encoder(face) + voice_encoder(voice) -> fusion -> heads.
    Gradients flow through face_encoder, voice_encoder, fusion_block, and heads.
    """

    def __init__(
        self,
        audio_input_dim: int,
        face_backbone: str = "efficientnet_b0",
        freeze_encoders: bool = False,
        dropout: float = None,
    ):
        super().__init__()
        dropout = dropout if dropout is not None else FUSION_DROPOUT
        self._audio_flat_dim = audio_input_dim
        self._n_mels = N_MELS
        self._audio_time = audio_input_dim // (2 * self._n_mels)

        face_full = FaceAgeModel(backbone=face_backbone, pretrained=False)
        self.face_encoder = nn.Sequential(
            OrderedDict([
                ("backbone", face_full.backbone),
                ("feature_processor", face_full.feature_processor),
            ])
        )

        self.voice_encoder = VoiceAgeModel(input_channels=2)

        self.face_norm = nn.LayerNorm(FACE_FEAT_DIM)
        self.voice_norm = nn.LayerNorm(VOICE_FEAT_DIM)
        self.voice_proj = nn.Linear(VOICE_FEAT_DIM, FACE_FEAT_DIM)
        FUSION_INPUT_DIM = FACE_FEAT_DIM + FACE_FEAT_DIM

        self.fusion_block = nn.Sequential(
            nn.Linear(FUSION_INPUT_DIM, FUSION_HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.age_head = nn.Linear(FUSION_HIDDEN_DIM, 1)
        self.age_group_head = nn.Linear(FUSION_HIDDEN_DIM, 4)
        self.adult_head = nn.Linear(FUSION_HIDDEN_DIM, 1)
        self._no_fusion_proj = nn.Linear(FUSION_INPUT_DIM, FUSION_HIDDEN_DIM)

        if freeze_encoders:
            for p in self.face_encoder.parameters():
                p.requires_grad = False
            for p in self.voice_encoder.parameters():
                p.requires_grad = False

    def forward(
        self,
        face_input: torch.Tensor,
        voice_input: torch.Tensor,
        ablation: str = None,
    ):
        B = voice_input.size(0)
        voice_2d = voice_input.view(B, 2, self._n_mels, self._audio_time)
        f_face = self.face_encoder(face_input)
        out_voice = self.voice_encoder(voice_2d, return_features=True)
        f_voice = out_voice["features"]
        f_face = self.face_norm(f_face)
        f_voice = self.voice_norm(f_voice)
        f_voice = self.voice_proj(f_voice)
        if ablation == "face_only":
            f_voice = torch.zeros_like(f_voice, device=f_voice.device, dtype=f_voice.dtype)
        elif ablation == "voice_only":
            f_face = torch.zeros_like(f_face, device=f_face.device, dtype=f_face.dtype)
        joint = torch.cat([f_face, f_voice], dim=-1)
        if ablation == "no_fusion":
            fused = joint
            if hasattr(self, "_no_fusion_proj") and self._no_fusion_proj is not None:
                fused = self._no_fusion_proj(fused)
        else:
            fused = self.fusion_block(joint)
        age = self.age_head(fused).squeeze(-1).clamp(0, 100)
        age_group_logits = self.age_group_head(fused)
        adult_logit = self.adult_head(fused).squeeze(-1)
        return {
            "age": age,
            "age_group_logits": age_group_logits,
            "is_adult_logit": adult_logit,
        }


def create_fusion_model(
    audio_input_dim: int,
    face_backbone: str = None,
    freeze_encoders: bool = False,
    device=DEVICE,
):
    face_backbone = face_backbone or FACE_BACKBONE
    model = MultimodalFusionModel(
        audio_input_dim=audio_input_dim,
        face_backbone=face_backbone,
        freeze_encoders=freeze_encoders,
    )
    return model.to(device)


def load_fusion_model(checkpoint_path: str, device=DEVICE):
    from loguru import logger
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    audio_dim = ckpt.get("audio_dim")
    if audio_dim is None:
        max_frames = int(MAX_AUDIO_LENGTH * SAMPLE_RATE / HOP_LENGTH)
        audio_dim = 2 * N_MELS * max_frames
        logger.warning("Checkpoint missing 'audio_dim'; using default from config: %s", audio_dim)
    model = MultimodalFusionModel(audio_input_dim=audio_dim, freeze_encoders=False)
    state = ckpt["model_state_dict"]
    missing, _ = model.load_state_dict(state, strict=False)
    if any("voice_proj" in k for k in missing):
        logger.warning(
            "Old checkpoint without voice_proj: initializing voice projection randomly. "
            "Retrain or save a new checkpoint for full fusion stability."
        )
    model = model.to(device)
    model.eval()
    return model

"""
ImageBind wrapper for age authentication (comparison only).

This model uses a frozen pretrained ImageBind encoder.
Only the prediction head is trained.

ImageBind was NOT trained for age prediction; no age supervision exists
in pretraining. Results are for comparison and limitation analysis only.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Union, Optional
from loguru import logger

# Optional: ImageBind is not in base requirements; install from https://github.com/facebookresearch/ImageBind
try:
    from imagebind.models import imagebind_model
    from imagebind.models.imagebind_model import ModalityType
    from imagebind import data as imagebind_data
    IMAGEBIND_AVAILABLE = True
except ImportError:
    imagebind_model = None
    ModalityType = None
    imagebind_data = None
    IMAGEBIND_AVAILABLE = False

import sys
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from src.config import DEVICE

# Embedding dimension from imagebind_huge
IMAGEBIND_EMBED_DIM = 1024


class ImageBindAgeWrapper(nn.Module):
    """
    Wrapper: frozen ImageBind (image + audio) -> joint embedding -> MLP head for age.

    This model uses a frozen pretrained ImageBind encoder. Only the prediction
    head is trained. ImageBind was NOT trained for age prediction.
    """

    def __init__(
        self,
        encoder=None,
        embed_dim: int = IMAGEBIND_EMBED_DIM,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        if not IMAGEBIND_AVAILABLE:
            raise ImportError(
                "ImageBind is required. Install from: https://github.com/facebookresearch/ImageBind"
            )
        self.encoder = encoder or imagebind_model.imagebind_huge(pretrained=True)
        self.embed_dim = embed_dim
        # Joint: concat vision + audio embeddings -> 2 * embed_dim
        fusion_dim = 2 * embed_dim
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            logger.info("ImageBindAgeWrapper: froze ImageBind encoder")

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        head_out = hidden_dim // 2
        self.age_head = nn.Linear(head_out, 1)
        self.age_group_head = nn.Linear(head_out, 4)
        self.adult_head = nn.Linear(head_out, 1)

    def forward(
        self,
        image_paths: Optional[List[str]] = None,
        audio_paths: Optional[List[str]] = None,
        image_tensors: Optional[torch.Tensor] = None,
        audio_tensors: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Forward pass. Provide either paths (for ImageBind loading) or pre-loaded tensors.

        Returns dict: age, age_group_logits, is_adult_logit, is_adult (same as baselines).
        """
        dev = device or next(self.parameters()).device
        if image_paths is not None and audio_paths is not None:
            # Official ImageBind data loading from paths
            inputs = {
                ModalityType.VISION: imagebind_data.load_and_transform_vision_data(
                    image_paths, dev
                ),
                ModalityType.AUDIO: imagebind_data.load_and_transform_audio_data(
                    audio_paths, dev
                ),
            }
        elif image_tensors is not None and audio_tensors is not None:
            # Pre-loaded tensors (e.g. from custom dataloader that matches ImageBind format)
            inputs = {
                ModalityType.VISION: image_tensors.to(dev),
                ModalityType.AUDIO: audio_tensors.to(dev),
            }
        else:
            raise ValueError("Provide either (image_paths, audio_paths) or (image_tensors, audio_tensors)")

        with torch.set_grad_enabled(self.training and any(p.requires_grad for p in self.encoder.parameters())):
            embeds = self.encoder(inputs)
        vision_emb = embeds[ModalityType.VISION]   # (B, 1024)
        audio_emb = embeds[ModalityType.AUDIO]     # (B, 1024)
        joint = torch.cat([vision_emb, audio_emb], dim=-1)  # (B, 2048)

        h = self.head(joint)
        age = self.age_head(h).squeeze(-1).clamp(0, 100)
        age_group_logits = self.age_group_head(h)
        adult_logit = self.adult_head(h).squeeze(-1)

        return {
            "age": age,
            "age_group_logits": age_group_logits,
            "is_adult_logit": adult_logit,
            "is_adult": torch.sigmoid(adult_logit),
        }


def create_imagebind_age_model(
    pretrained: bool = True,
    freeze_encoder: bool = True,
    device: torch.device = DEVICE,
):
    """Create ImageBind age wrapper with frozen encoder and trainable head."""
    if not IMAGEBIND_AVAILABLE:
        raise ImportError("ImageBind not installed. See https://github.com/facebookresearch/ImageBind")
    encoder = imagebind_model.imagebind_huge(pretrained=pretrained)
    model = ImageBindAgeWrapper(
        encoder=encoder,
        freeze_encoder=freeze_encoder,
    )
    return model.to(device)

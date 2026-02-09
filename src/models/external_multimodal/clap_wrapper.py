"""
CLAP wrapper for age authentication (comparison only).

CLAP is evaluated as an existing pretrained multimodal model (audio-language)
adapted for age prediction. The backbone is frozen; only a shallow MLP head is trained.

These models were NOT trained for age prediction; no age supervision exists
in pretraining. Results are for comparison and limitation analysis only.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Optional
from loguru import logger

# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
import sys
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import DEVICE

# ---------------------------------------------------------------------
# Optional CLAP dependencies
# ---------------------------------------------------------------------
try:
    from transformers import ClapModel, ClapProcessor
    import librosa
    CLAP_AVAILABLE = True
except ImportError:
    ClapModel = None
    ClapProcessor = None
    librosa = None
    CLAP_AVAILABLE = False

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
CLAP_AUDIO_EMBED_DIM = 512
CLAP_SAMPLE_RATE = 48_000


class CLAPAgeWrapper(nn.Module):
    """
    Frozen CLAP audio encoder + trainable MLP head for age prediction.
    """

    def __init__(
        self,
        encoder=None,
        processor=None,
        embed_dim: int = CLAP_AUDIO_EMBED_DIM,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        freeze_encoder: bool = True,
        model_id: str = "laion/clap-htsat-unfused",
    ):
        super().__init__()

        if not CLAP_AVAILABLE:
            raise ImportError(
                "CLAP requires `transformers` and `librosa`. "
                "Install with: pip install transformers librosa"
            )

        self.model_id = model_id
        self.encoder = encoder or ClapModel.from_pretrained(model_id)
        self.processor = processor or ClapProcessor.from_pretrained(model_id)
        self.embed_dim = embed_dim

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            logger.info("CLAPAgeWrapper: froze CLAP encoder")

        # -----------------------------------------------------------------
        # Task head (trainable)
        # -----------------------------------------------------------------
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
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

    # ---------------------------------------------------------------------
    # Audio loading
    # ---------------------------------------------------------------------
    def _load_audio_batch(
        self,
        audio_paths: List[str],
        device: torch.device,
        max_length_sec: float = 10.0,
    ) -> torch.Tensor:
        """
        Load audio files and resample to CLAP sample rate.
        Returns tensor of shape (B, T).
        """
        waveforms = []
        max_samples = int(CLAP_SAMPLE_RATE * max_length_sec)

        for path in audio_paths:
            try:
                y, _ = librosa.load(path, sr=CLAP_SAMPLE_RATE, mono=True)

                if len(y) > max_samples:
                    y = y[:max_samples]
                elif len(y) < max_samples:
                    y = np.pad(y, (0, max_samples - len(y)), mode="constant")

                waveforms.append(y)

            except Exception as e:
                logger.warning(f"CLAP audio load failed for {path}: {e}")
                waveforms.append(np.zeros(max_samples, dtype=np.float32))

        return torch.tensor(
            np.stack(waveforms),
            dtype=torch.float32,
            device=device,
        )

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(
        self,
        audio_paths: Optional[List[str]] = None,
        audio_waveforms: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Forward pass.

        Provide either:
        - audio_paths: list[str]
        - audio_waveforms: Tensor (B, T) at 48 kHz

        Returns:
            dict with keys:
            - age
            - age_group_logits
            - is_adult_logit
            - is_adult
        """
        dev = device or next(self.parameters()).device

        if audio_paths is not None:
            waveforms = self._load_audio_batch(audio_paths, dev)
        elif audio_waveforms is not None:
            waveforms = audio_waveforms.to(dev)
            if waveforms.dim() == 1:
                waveforms = waveforms.unsqueeze(0)
        else:
            raise ValueError("Provide either audio_paths or audio_waveforms")

        # CLAP processor expects list of numpy arrays
        audio_list = [waveforms[i].cpu().numpy() for i in range(waveforms.shape[0])]
        inputs = self.processor(
            audio=audio_list,
            sampling_rate=CLAP_SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(dev) for k, v in inputs.items()}

        # -----------------------------------------------------------------
        # Extract audio embeddings correctly (IMPORTANT)
        # -----------------------------------------------------------------
        with torch.set_grad_enabled(self.training and any(p.requires_grad for p in self.encoder.parameters())):
            outputs = self.encoder.get_audio_features(**inputs)

            # Handle different Transformers versions
            if isinstance(outputs, torch.Tensor):
                embeds = outputs
            elif hasattr(outputs, "audio_embeds"):
                embeds = outputs.audio_embeds
            elif hasattr(outputs, "pooler_output"):
                embeds = outputs.pooler_output
            else:
                raise TypeError(f"Unexpected CLAP output type: {type(outputs)}")


        # -----------------------------------------------------------------
        # Task heads
        # -----------------------------------------------------------------
        h = self.head(embeds)

        age = self.age_head(h).squeeze(-1).clamp(0, 100)
        age_group_logits = self.age_group_head(h)
        adult_logit = self.adult_head(h).squeeze(-1)

        return {
            "age": age,
            "age_group_logits": age_group_logits,
            "is_adult_logit": adult_logit,
            "is_adult": torch.sigmoid(adult_logit),
        }


# -------------------------------------------------------------------------
# Factory
# -------------------------------------------------------------------------
def create_clap_age_model(
    model_id: str = "laion/clap-htsat-unfused",
    freeze_encoder: bool = True,
    device: torch.device = DEVICE,
):
    """
    Create CLAP age wrapper with frozen encoder and trainable head.
    """
    if not CLAP_AVAILABLE:
        raise ImportError("transformers and librosa required for CLAP")

    encoder = ClapModel.from_pretrained(model_id)
    processor = ClapProcessor.from_pretrained(model_id)

    model = CLAPAgeWrapper(
        encoder=encoder,
        processor=processor,
        freeze_encoder=freeze_encoder,
        model_id=model_id,
    )
    return model.to(device)

"""
Phase 3 — Embedding MLP and binary head for cluster-age verification.
Binary head is sole classification authority; single threshold θ applied to P(minor).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.cluster_age.features import FEATURE_DIM

EMBEDDING_DIM = 64
INPUT_DIM = FEATURE_DIM  # 128


class EmbeddingMLP(nn.Module):
    """
    Embedding MLP: 128 -> 256 -> 128 -> 64.
    ReLU + Dropout(0.2) after hidden layers.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dims: tuple[int, int] = (256, 128),
        output_dim: int = EMBEDDING_DIM,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BinaryHead(nn.Module):
    """
    Binary head: input 64 -> output 1 logit (minor).
    P(minor) = sigmoid(logit). Sole classification authority.
    """

    def __init__(self, input_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)

"""
Phase 2 — Spoof gate interface.
Runs before age path; if failed, API must NOT return classification.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class SpoofGateBase(ABC):
    """
    Interface for spoof/liveness detection on waveform.
    Must not use age or cluster; only signal quality / manipulation detection.
    """

    @abstractmethod
    def run(self, waveform: np.ndarray, sample_rate: int) -> Tuple[bool, float]:
        """
        Returns (passed, score).
        - passed: True if the request is allowed to proceed to age path.
        - score: float, interpretation is implementation-defined (e.g. bonafide probability).
        If passed is False, the API must not return classification.
        """
        pass


class SpoofGateStub(SpoofGateBase):
    """
    Stub implementation: always passes. Replace with real spoof detector later.
    Keeps pipeline testable without a trained spoof model.
    """

    def run(self, waveform: np.ndarray, sample_rate: int) -> Tuple[bool, float]:
        if waveform is None or len(waveform) == 0:
            return False, 0.0
        return True, 1.0

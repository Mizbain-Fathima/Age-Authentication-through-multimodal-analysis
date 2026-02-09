"""
Existing pretrained multimodal model wrappers for comparison and analysis.

These models were NOT trained for age prediction; no age supervision exists
in their pretraining. Results are for comparison and limitation analysis only.
"""

from src.models.external_multimodal.imagebind_wrapper import (
    ImageBindAgeWrapper,
    create_imagebind_age_model,
)
from src.models.external_multimodal.clap_wrapper import (
    CLAPAgeWrapper,
    create_clap_age_model,
)

__all__ = [
    "ImageBindAgeWrapper",
    "create_imagebind_age_model",
    "CLAPAgeWrapper",
    "create_clap_age_model",
]

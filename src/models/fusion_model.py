"""
Multimodal Fusion Model
Combines face and voice features for robust age prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import FUSION_HIDDEN_DIM, FUSION_DROPOUT, DEVICE
from src.models.face_model import FaceAgeModel
from src.models.voice_model import VoiceAgeModel


class GatedFusion(nn.Module):
    """Gated fusion mechanism for combining modalities"""
    
    def __init__(self, face_dim, voice_dim, hidden_dim):
        super().__init__()
        self.face_transform = nn.Linear(face_dim, hidden_dim)
        self.voice_transform = nn.Linear(voice_dim, hidden_dim)
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
    
    def forward(self, face_features, voice_features):
        face_h = self.face_transform(face_features)
        voice_h = self.voice_transform(voice_features)
        
        # Concatenate and compute gate
        concat = torch.cat([face_h, voice_h], dim=-1)
        gate = self.gate(concat)
        
        # Gated combination
        fused = gate * face_h + (1 - gate) * voice_h
        return fused


class CrossModalAttention(nn.Module):
    """Cross-modal attention for face-voice interaction"""
    
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.face_to_voice = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.voice_to_face = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, face_features, voice_features):
        # Face attending to voice
        face_attended, _ = self.face_to_voice(
            face_features.unsqueeze(1),
            voice_features.unsqueeze(1),
            voice_features.unsqueeze(1)
        )
        face_out = self.norm1(face_features + face_attended.squeeze(1))
        
        # Voice attending to face
        voice_attended, _ = self.voice_to_face(
            voice_features.unsqueeze(1),
            face_features.unsqueeze(1),
            face_features.unsqueeze(1)
        )
        voice_out = self.norm2(voice_features + voice_attended.squeeze(1))
        
        return face_out, voice_out


class MultimodalFusionModel(nn.Module):
    """
    Multimodal age prediction model combining face and voice
    
    Architecture:
        - Face encoder (frozen or fine-tuned)
        - Voice encoder (frozen or fine-tuned)
        - Cross-modal attention
        - Gated fusion
        - Multi-task prediction heads
    """
    
    def __init__(self, face_model=None, voice_model=None, 
                 hidden_dim=FUSION_HIDDEN_DIM, dropout=FUSION_DROPOUT,
                 freeze_encoders=True):
        super().__init__()
        
        # Load or create encoders
        if face_model is None:
            self.face_encoder = FaceAgeModel(pretrained=True)
        else:
            self.face_encoder = face_model
        
        if voice_model is None:
            self.voice_encoder = VoiceAgeModel()
        else:
            self.voice_encoder = voice_model
        
        # Get feature dimensions
        face_dim = self.face_encoder.get_feature_dim()
        voice_dim = self.voice_encoder.get_feature_dim()
        
        # Freeze encoders if specified
        if freeze_encoders:
            for param in self.face_encoder.parameters():
                param.requires_grad = False
            for param in self.voice_encoder.parameters():
                param.requires_grad = False
            logger.info("Froze face and voice encoder parameters")
        
        # Feature projection to common dimension
        self.face_projection = nn.Sequential(
            nn.Linear(face_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        self.voice_projection = nn.Sequential(
            nn.Linear(voice_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Cross-modal attention
        self.cross_attention = CrossModalAttention(hidden_dim, num_heads=4)
        
        # Gated fusion
        self.gated_fusion = GatedFusion(hidden_dim, hidden_dim, hidden_dim)
        
        # Additional fusion layers
        self.fusion_processor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),  # concat + gated
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Confidence estimation for each modality
        self.face_confidence = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.voice_confidence = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Final prediction heads
        self.age_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        
        self.age_group_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 4)
        )
        
        self.adult_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1)
        )
        
        logger.info(f"Initialized MultimodalFusionModel with hidden_dim={hidden_dim}")
    
    def forward(self, face_input, voice_input, return_details=False):
        """
        Forward pass
        
        Args:
            face_input: Face images (B, 3, H, W)
            voice_input: Audio features (B, C, H, W)
            return_details: Whether to return detailed outputs
        
        Returns:
            Dictionary with fused predictions
        """
        # Extract features from encoders
        face_out = self.face_encoder(face_input, return_features=True)
        voice_out = self.voice_encoder(voice_input, return_features=True)
        
        face_features = face_out['features']  # (B, face_dim)
        voice_features = voice_out['features']  # (B, voice_dim)
        
        # Project to common dimension
        face_proj = self.face_projection(face_features)
        voice_proj = self.voice_projection(voice_features)
        
        # Cross-modal attention
        face_attended, voice_attended = self.cross_attention(face_proj, voice_proj)
        
        # Gated fusion
        gated_features = self.gated_fusion(face_attended, voice_attended)
        
        # Concatenate different fusion strategies
        concat_features = torch.cat([face_attended, voice_attended, gated_features], dim=-1)
        
        # Process fused features
        fused_features = self.fusion_processor(concat_features)
        
        # Modality confidence scores
        face_conf = self.face_confidence(face_proj)
        voice_conf = self.voice_confidence(voice_proj)
        
        # Final predictions
        age = self.age_head(fused_features).squeeze(-1)
        age = torch.clamp(age, 0, 100)
        
        age_group_logits = self.age_group_head(fused_features)
        
        adult_logit = self.adult_head(fused_features).squeeze(-1)
        
        output = {
            'age': age,
            'age_group_logits': age_group_logits,
            'age_group': torch.argmax(age_group_logits, dim=-1),
            'is_adult_logit': adult_logit,
            'is_adult': torch.sigmoid(adult_logit),
            'is_adult_binary': (torch.sigmoid(adult_logit) > 0.5).float(),
            'face_confidence': face_conf.squeeze(-1),
            'voice_confidence': voice_conf.squeeze(-1)
        }
        
        if return_details:
            output['face_age'] = face_out['age']
            output['voice_age'] = voice_out['age']
            output['face_features'] = face_features
            output['voice_features'] = voice_features
            output['fused_features'] = fused_features
        
        return output
    
    def predict_single_modality(self, input_data, modality='face'):
        """Predict using only one modality"""
        if modality == 'face':
            return self.face_encoder(input_data, return_features=True)
        elif modality == 'voice':
            return self.voice_encoder(input_data, return_features=True)
        else:
            raise ValueError(f"Unknown modality: {modality}")


class FusionLoss(nn.Module):
    """Loss function for multimodal fusion model"""
    
    def __init__(self, age_weight=1.0, group_weight=0.5, adult_weight=0.5,
                 consistency_weight=0.2, class_weights=None):
        super().__init__()
        
        self.age_weight = age_weight
        self.group_weight = group_weight
        self.adult_weight = adult_weight
        self.consistency_weight = consistency_weight
        
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, predictions, targets, face_pred=None, voice_pred=None):
        """
        Calculate fusion loss with optional consistency regularization
        """
        # Main task losses
        age_loss = self.mae_loss(predictions['age'], targets['age']) + \
                   0.5 * self.mse_loss(predictions['age'], targets['age'])
        
        group_loss = self.ce_loss(predictions['age_group_logits'], targets['age_group'])
        
        adult_loss = self.bce_loss(predictions['is_adult_logit'], targets['is_adult'])
        
        total_loss = (
            self.age_weight * age_loss +
            self.group_weight * group_loss +
            self.adult_weight * adult_loss
        )
        
        losses = {
            'total': total_loss,
            'age': age_loss,
            'group': group_loss,
            'adult': adult_loss
        }
        
        # Optional consistency loss between modalities
        if face_pred is not None and voice_pred is not None:
            consistency_loss = self.mae_loss(face_pred['age'], voice_pred['age'])
            total_loss = total_loss + self.consistency_weight * consistency_loss
            losses['consistency'] = consistency_loss
            losses['total'] = total_loss
        
        return losses


def create_fusion_model(face_checkpoint=None, voice_checkpoint=None, 
                       freeze_encoders=True, device=DEVICE):
    """
    Factory function to create fusion model
    
    Args:
        face_checkpoint: Path to face model checkpoint
        voice_checkpoint: Path to voice model checkpoint
        freeze_encoders: Whether to freeze encoder weights
        device: Target device
    """
    # Load pretrained encoders if checkpoints provided
    face_model = None
    voice_model = None
    
    if face_checkpoint:
        from src.models.face_model import load_face_model
        face_model = load_face_model(face_checkpoint, device=device)
    
    if voice_checkpoint:
        from src.models.voice_model import load_voice_model
        voice_model = load_voice_model(voice_checkpoint, device=device)
    
    model = MultimodalFusionModel(
        face_model=face_model,
        voice_model=voice_model,
        freeze_encoders=freeze_encoders
    )
    model = model.to(device)
    
    return model


def load_fusion_model(checkpoint_path, device=DEVICE):
    """Load complete fusion model from checkpoint"""
    model = MultimodalFusionModel(freeze_encoders=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    logger.info(f"Loaded fusion model from {checkpoint_path}")
    return model


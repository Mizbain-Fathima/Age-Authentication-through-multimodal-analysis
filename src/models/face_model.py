"""
Face Age Prediction Model
CNN-based model for predicting age from face images using transfer learning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from loguru import logger

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import FACE_DROPOUT, FACE_BACKBONE, DEVICE


class AttentionBlock(nn.Module):
    """Spatial attention mechanism for face features"""
    
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.BatchNorm2d(channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights


class FaceAgeModel(nn.Module):
    """
    Multi-task face age prediction model
    
    Outputs:
        - age: Continuous age regression (0-100)
        - age_group: Classification (child/teen/adult/senior)
        - is_adult: Binary classification (18+)
        - features: Intermediate features for fusion
    """
    
    def __init__(self, backbone='efficientnet_b0', dropout=FACE_DROPOUT, pretrained=True):
        super().__init__()
        
        self.backbone_name = backbone
        
        # Load pretrained backbone
        if backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == 'efficientnet_b2':
            self.backbone = models.efficientnet_b2(weights='IMAGENET1K_V1' if pretrained else None)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        self.feature_dim = feature_dim
        
        # Feature processing
        self.feature_processor = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Age regression head
        self.age_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1)
        )
        
        # Age group classification head (4 classes)
        self.age_group_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 4)
        )
        
        # Binary adult classification head
        self.adult_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1)
        )
        
        logger.info(f"Initialized FaceAgeModel with {backbone} backbone, feature_dim={feature_dim}")
    
    def forward(self, x, return_features=False):
        """
        Forward pass
        
        Args:
            x: Input images (B, 3, H, W)
            return_features: Whether to return intermediate features
        
        Returns:
            Dictionary with predictions and optionally features
        """
        # Extract backbone features
        backbone_features = self.backbone(x)
        
        # Process features
        features = self.feature_processor(backbone_features)
        
        # Task-specific predictions
        age = self.age_head(features).squeeze(-1)
        age = torch.clamp(age, 0, 100)  # Clamp to valid range
        
        age_group_logits = self.age_group_head(features)
        
        adult_logit = self.adult_head(features).squeeze(-1)
        
        output = {
            'age': age,
            'age_group_logits': age_group_logits,
            'is_adult_logit': adult_logit,
            'is_adult': torch.sigmoid(adult_logit)
        }
        
        if return_features:
            output['features'] = features
        
        return output
    
    def get_feature_dim(self):
        """Return the dimension of extracted features"""
        return 256


class FaceAgeLoss(nn.Module):
    """Multi-task loss for face age prediction"""
    
    def __init__(self, age_weight=1.0, group_weight=0.5, adult_weight=0.5, 
                 class_weights=None):
        super().__init__()
        
        self.age_weight = age_weight
        self.group_weight = group_weight
        self.adult_weight = adult_weight
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, predictions, targets):
        """
        Calculate multi-task loss
        
        Args:
            predictions: Model output dictionary
            targets: Dictionary with 'age', 'age_group', 'is_adult'
        """
        # Age regression loss (MAE + MSE)
        age_loss = self.mae_loss(predictions['age'], targets['age']) + \
                   0.5 * self.mse_loss(predictions['age'], targets['age'])
        
        # Age group classification loss
        group_loss = self.ce_loss(predictions['age_group_logits'], targets['age_group'])
        
        # Adult binary classification loss
        adult_loss = self.bce_loss(predictions['is_adult_logit'], targets['is_adult'])
        
        # Total weighted loss
        total_loss = (
            self.age_weight * age_loss +
            self.group_weight * group_loss +
            self.adult_weight * adult_loss
        )
        
        return {
            'total': total_loss,
            'age': age_loss,
            'group': group_loss,
            'adult': adult_loss
        }


def create_face_model(backbone='efficientnet_b0', pretrained=True, device=DEVICE):
    """Factory function to create face model"""
    model = FaceAgeModel(backbone=backbone, pretrained=pretrained)
    model = model.to(device)
    return model


def load_face_model(checkpoint_path, backbone='efficientnet_b0', device=DEVICE):
    """Load face model from checkpoint"""
    model = FaceAgeModel(backbone=backbone, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    logger.info(f"Loaded face model from {checkpoint_path}")
    return model


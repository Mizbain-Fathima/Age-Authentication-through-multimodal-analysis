"""
Voice Age Prediction Model - Optimized for CPU Training
Lightweight CNN-based model for predicting age from audio features (MFCC + Mel Spectrogram)
Supports all age groups including children (5-12 years)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import VOICE_DROPOUT, N_MELS, DEVICE, AGE_GROUPS


class LightConvBlock(nn.Module):
    """Lightweight convolutional block with BatchNorm and ReLU"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution - much faster than standard conv"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return F.relu(self.bn(x), inplace=True)


class SqueezeExcitation(nn.Module):
    """Lightweight channel attention"""
    
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
    
    def forward(self, x):
        # x: (B, C, H, W)
        b, c, _, _ = x.size()
        y = x.mean(dim=(2, 3))  # Global average pooling
        y = F.relu(self.fc1(y), inplace=True)
        y = torch.sigmoid(self.fc2(y))
        return x * y.view(b, c, 1, 1)


class FrequencyAttention(nn.Module):
    """
    Attention module that focuses on different frequency bands
    Helps distinguish children's higher-pitched voices from adults
    """
    
    def __init__(self, freq_dim, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(freq_dim, freq_dim // reduction)
        self.fc2 = nn.Linear(freq_dim // reduction, freq_dim)
    
    def forward(self, x):
        # x: (B, C, H, W) where H is frequency dimension
        b, c, h, w = x.size()
        # Pool over time and channels to get frequency importance
        y = x.mean(dim=(1, 3))  # (B, H)
        y = F.relu(self.fc1(y), inplace=True)
        y = torch.sigmoid(self.fc2(y))
        return x * y.view(b, 1, h, 1)


class VoiceAgeModel(nn.Module):
    """
    Lightweight Voice age prediction model optimized for CPU
    Supports all age groups including children (5-12 years)
    
    Architecture:
        - Efficient CNN encoder with depthwise separable convolutions
        - Squeeze-and-excitation attention (channel-wise)
        - Frequency attention (for capturing pitch differences between children/adults)
        - Global pooling (no LSTM for speed)
        - Multi-task heads
    
    Children's voices have distinct characteristics:
        - Higher fundamental frequency (pitch)
        - Different formant frequencies
        - Shorter vocal tract = higher resonant frequencies
    """
    
    def __init__(self, input_channels=2, dropout=VOICE_DROPOUT, num_age_groups=4):
        super().__init__()
        
        self.num_age_groups = num_age_groups
        
        # Lightweight CNN encoder with frequency attention for children's voices
        # Block 0: Initial processing
        self.conv_init = LightConvBlock(input_channels, 16, 3, 1, 1)
        self.pool0 = nn.MaxPool2d(2, 2)
        self.drop0 = nn.Dropout2d(dropout * 0.25)
        
        # Block 1: Depthwise separable with channel attention
        self.conv1 = DepthwiseSeparableConv(16, 32, 3, 1, 1)
        self.se1 = SqueezeExcitation(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(dropout * 0.25)
        
        # Block 2: With frequency attention (helps distinguish children's higher pitch)
        self.conv2 = DepthwiseSeparableConv(32, 64, 3, 1, 1)
        self.se2 = SqueezeExcitation(64)
        self.freq_attn = FrequencyAttention(32, reduction=4)  # After pool, freq dim is ~32
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(dropout * 0.5)
        
        # Block 3: Final encoding
        self.conv3 = DepthwiseSeparableConv(64, 128, 3, 1, 1)
        self.se3 = SqueezeExcitation(128)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature dimension after encoder
        self.feature_dim = 128
        
        # Feature processor with increased capacity for age diversity
        self.feature_processor = nn.Sequential(
            nn.Linear(128, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(96, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5)
        )
        
        # Age regression head
        self.age_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )
        
        # Age group classification head (4 classes: child, teen, adult, senior)
        self.age_group_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_age_groups)
        )
        
        # Binary adult classification head (child/teen vs adult/senior)
        self.adult_head = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1)
        )
        
        # Initialize weights
        self._init_weights()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"Initialized VoiceAgeModel (Lite): {trainable_params:,} trainable params")
    
    def _init_weights(self):
        """Initialize weights for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)
    
    def forward(self, x, return_features=False):
        """
        Forward pass
        
        Args:
            x: Audio features (B, C, H, W) - typically (B, 2, 128, T)
            return_features: Whether to return intermediate features
        
        Returns:
            Dictionary with predictions and optionally features
        """
        batch_size = x.size(0)
        
        # Block 0: Initial processing
        x = self.conv_init(x)
        x = self.pool0(x)
        x = self.drop0(x)
        
        # Block 1
        x = self.conv1(x)
        x = self.se1(x)
        x = self.pool1(x)
        x = self.drop1(x)
        
        # Block 2 with frequency attention
        x = self.conv2(x)
        x = self.se2(x)
        x = self.freq_attn(x)  # Attend to frequency bands (helps with children's pitch)
        x = self.pool2(x)
        x = self.drop2(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.se3(x)
        features = self.global_pool(x)  # (B, 128, 1, 1)
        features = features.view(batch_size, -1)  # (B, 128)
        
        # Feature processing
        features = self.feature_processor(features)  # (B, 64)
        
        # Task-specific predictions
        age = self.age_head(features).squeeze(-1)
        age = torch.clamp(age, 0, 100)
        
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
        """Return the dimension of extracted features for fusion"""
        return 64


class VoiceAgeLoss(nn.Module):
    """Multi-task loss for voice age prediction"""
    
    def __init__(self, age_weight=1.0, group_weight=0.3, adult_weight=0.3,
                 class_weights=None):
        super().__init__()
        
        self.age_weight = age_weight
        self.group_weight = group_weight
        self.adult_weight = adult_weight
        
        # Use Smooth L1 loss for better gradient behavior
        self.age_loss = nn.SmoothL1Loss()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, predictions, targets):
        """Calculate multi-task loss"""
        # Age regression loss (Smooth L1 is more robust to outliers)
        age_loss = self.age_loss(predictions['age'], targets['age'])
        
        # Age group classification loss
        group_loss = self.ce_loss(predictions['age_group_logits'], targets['age_group'])
        
        # Adult binary classification loss
        adult_loss = self.bce_loss(predictions['is_adult_logit'], targets['is_adult'])
        
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


def create_voice_model(input_channels=2, device=DEVICE):
    """Factory function to create voice model"""
    model = VoiceAgeModel(input_channels=input_channels)
    model = model.to(device)
    return model


def load_voice_model(checkpoint_path, input_channels=2, device=DEVICE):
    """Load voice model from checkpoint"""
    model = VoiceAgeModel(input_channels=input_channels)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    logger.info(f"Loaded voice model from {checkpoint_path}")
    return model

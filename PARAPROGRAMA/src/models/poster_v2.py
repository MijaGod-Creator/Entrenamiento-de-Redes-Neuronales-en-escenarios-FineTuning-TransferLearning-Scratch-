from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from src.models.qcs import InceptionResnetV1Features


class MultiheadAttentionBlock(nn.Module):
    """
    Standard Multi-head Attention Block with Residual Connection and Layer Normalization.
    """
    def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # q, k, v are batch_first: (B, SeqLen, E)
        attn_out, _ = self.mha(q, k, v)
        return self.norm(q + self.dropout(attn_out))


class PosterV2(nn.Module):
    """
    PosterV2 (POSTER++) architecture.
    Supports standard torchvision backbones and face-pretrained InceptionResnetV1.
    Uses an implicit landmark query stream to bidirectionally fuse visual and geometric face features.
    """
    def __init__(self, num_classes: int, backbone_name: str = "inception_resnet_v1", d_model: int = 256, num_landmarks: int = 68, dropout: float = 0.35):
        super().__init__()
        self.d_model = d_model
        self.num_landmarks = num_landmarks
        
        # Load pre-trained backbone
        if backbone_name == "resnet18":
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            in_features = 512
            self.backbone = nn.Sequential(
                base.conv1, base.bn1, base.relu, base.maxpool,
                base.layer1, base.layer2, base.layer3, base.layer4
            )
        elif backbone_name == "resnet50":
            base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            in_features = 2048
            self.backbone = nn.Sequential(
                base.conv1, base.bn1, base.relu, base.maxpool,
                base.layer1, base.layer2, base.layer3, base.layer4
            )
        elif backbone_name == "inception_resnet_v1":
            self.backbone = InceptionResnetV1Features(pretrained='vggface2')
            in_features = 1792
        else:
            raise ValueError(f"Backbone {backbone_name} no soportado para PosterV2 en esta implementación.")
        
        # Projection layer to d_model
        self.proj = nn.Conv2d(in_features, d_model, kernel_size=1)
        
        # Implicit landmark query embeddings (B, NumLandmarks, D)
        self.landmark_queries = nn.Parameter(torch.randn(1, num_landmarks, d_model))
        
        # Bidirectional Attention blocks
        # 1. Image-to-Landmark Attention (refines landmark features using image context)
        self.img_to_landmark = MultiheadAttentionBlock(d_model, nhead=8, dropout=0.1)
        # 2. Landmark-to-Image Attention (injects refined geometric landmark features back to image features)
        self.landmark_to_img = MultiheadAttentionBlock(d_model, nhead=8, dropout=0.1)
        
        # Global Avg Pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        # Feature extraction from backbone
        features = self.backbone(x)     # B, in_features, H, W
        features = self.proj(features)   # B, d_model, H, W
        
        H, W = features.shape[2], features.shape[3]
        
        # Flatten spatial dimensions for Transformer blocks
        img_seq = features.flatten(2).transpose(1, 2)  # B, H*W, d_model
        
        # Expand implicit landmark queries to batch size
        lm_seq = self.landmark_queries.expand(B, -1, -1)  # B, 68, d_model
        
        # Step 1: Image features refine landmarks (img_seq acts as keys and values)
        lm_refined = self.img_to_landmark(lm_seq, img_seq, img_seq)  # B, 68, d_model
        
        # Step 2: Landmark features refine image features (lm_refined acts as keys and values)
        img_refined = self.landmark_to_img(img_seq, lm_refined, lm_refined)  # B, H*W, d_model
        
        # Reshape refined features back to spatial map
        img_feat = img_refined.transpose(1, 2).view(B, self.d_model, H, W)
        
        # Global average pool and classify
        pooled = self.pool(img_feat).squeeze(-1).squeeze(-1)  # B, d_model
        return self.fc(pooled)

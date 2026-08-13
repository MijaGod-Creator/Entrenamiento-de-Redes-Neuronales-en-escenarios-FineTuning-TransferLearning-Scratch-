from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


class InceptionResnetV1Features(nn.Module):
    """
    Feature extractor wrapper for InceptionResnetV1 from facenet-pytorch.
    Pre-trained on VGGFace2 dataset for face-specific representation.
    Automatically handles input renormalization on-the-fly to match the
    face-recognition network's expected distribution.
    """
    def __init__(self, pretrained: str = 'vggface2'):
        super().__init__()
        from facenet_pytorch import InceptionResnetV1
        self.base = InceptionResnetV1(pretrained=pretrained)
        # Buffers for on-the-fly image distribution correction
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert from ImageNet distribution back to [0, 1]
        x = x * self.imagenet_std + self.imagenet_mean
        # Convert to [-1, 1] range expected by VGGFace2 pre-trained weights
        x = (x - 0.5) / 0.5

        # Run forward pass up to block8 to extract spatial features
        x = self.base.conv2d_1a(x)
        x = self.base.conv2d_2a(x)
        x = self.base.conv2d_2b(x)
        x = self.base.maxpool_3a(x)
        x = self.base.conv2d_3b(x)
        x = self.base.conv2d_4a(x)
        x = self.base.conv2d_4b(x)
        x = self.base.repeat_1(x)
        x = self.base.mixed_6a(x)
        x = self.base.repeat_2(x)
        x = self.base.mixed_7a(x)
        x = self.base.repeat_3(x)
        x = self.base.block8(x)
        return x  # Output shape: (B, 1792, 5, 5)


class CrossSimilarityAttention(nn.Module):
    """
    Cross Similarity Attention (CSA) module.
    Calculates attention matching query vectors from one feature map (x)
    against key/value vectors from another feature map (y).
    """
    def __init__(self, in_channels: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.scale = (in_channels // num_heads) ** -0.5

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Project and flatten spatial dimensions
        q = self.q_proj(x).flatten(2).transpose(1, 2)  # B, N, C
        k = self.k_proj(y).flatten(2)                 # B, C, N
        v = self.v_proj(y).flatten(2).transpose(1, 2)  # B, N, C

        N = H * W
        head_dim = C // self.num_heads
        
        # Reshape for multi-head attention
        q = q.view(B, N, self.num_heads, head_dim).transpose(1, 2)  # B, h, N, d
        k = k.view(B, self.num_heads, head_dim, N)                  # B, h, d, N
        v = v.view(B, N, self.num_heads, head_dim).transpose(1, 2)  # B, h, N, d

        # Compute dot product attention
        attn = torch.matmul(q, k) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Weighted sum of values
        out = torch.matmul(attn, v)  # B, h, N, d
        out = out.transpose(1, 2).contiguous().view(B, N, C).transpose(1, 2).view(B, C, H, W)
        
        return self.out_proj(out) + x  # Residual connection


class QCSModel(nn.Module):
    """
    QCS (Quadruplet Cross Similarity) Model.
    Supports standard torchvision backbones and face-pretrained InceptionResnetV1.
    During training, it pairs features batch-wise to perform cross similarity.
    During evaluation/inference, it falls back to self-attention.
    """
    def __init__(self, num_classes: int, backbone_name: str = "inception_resnet_v1", num_heads: int = 8, dropout: float = 0.35):
        super().__init__()
        self.backbone_name = backbone_name
        
        # Build backbone from torchvision applications or facenet-pytorch
        if backbone_name == "resnet18":
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            in_channels = 512
            self.features = nn.Sequential(
                base.conv1,
                base.bn1,
                base.relu,
                base.maxpool,
                base.layer1,
                base.layer2,
                base.layer3,
                base.layer4
            )
        elif backbone_name == "resnet50":
            base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            in_channels = 2048
            self.features = nn.Sequential(
                base.conv1,
                base.bn1,
                base.relu,
                base.maxpool,
                base.layer1,
                base.layer2,
                base.layer3,
                base.layer4
            )
        elif backbone_name == "inception_resnet_v1":
            self.features = InceptionResnetV1Features(pretrained='vggface2')
            in_channels = 1792
        else:
            raise ValueError(f"Backbone {backbone_name} no soportado para QCS en esta implementación.")
        
        # CSA and pool blocks
        self.csa = CrossSimilarityAttention(in_channels, num_heads=num_heads)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)  # B, C, H, W
        
        if self.training:
            # During training, pair features of different samples in the batch
            # using a batch shift (roll) to simulate cross similarity.
            if feat.size(0) > 1:
                feat_shifted = torch.roll(feat, shifts=1, dims=0)
                feat_refined = self.csa(feat, feat_shifted)
            else:
                feat_refined = self.csa(feat, feat)
        else:
            # During evaluation, perform self-attention for consistent single-branch inference
            feat_refined = self.csa(feat, feat)
            
        pooled = self.pool(feat_refined)
        return self.classifier(pooled)

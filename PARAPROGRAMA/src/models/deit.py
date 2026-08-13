import torch
import torch.nn as nn
import timm

class DeiTFER(nn.Module):
    """
    DeiT (Data-efficient Image Transformer) adapted for 7-class Facial Expression Recognition (FER).
    Uses a timm VisionTransformer backbone and a custom linear classifier head with dropout.
    """
    def __init__(self, num_classes: int = 7, model_name: str = "deit_tiny_patch16_224", pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        # Load DeiT backbone (defaults to deit_tiny for mobility and fast training on 4GB VRAM)
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        
        # Determine the hidden embedding dimension
        embed_dim = self.backbone.num_features
        
        # Replace the original classification heads with Identity
        self.backbone.head = nn.Identity()
        if hasattr(self.backbone, "head_dist"):
            self.backbone.head_dist = nn.Identity()
            
        # Custom task-specific classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features from DeiT (returns class token state or average patch representation depending on configuration)
        features = self.backbone(x)
        # Classify
        logits = self.classifier(features)
        return logits

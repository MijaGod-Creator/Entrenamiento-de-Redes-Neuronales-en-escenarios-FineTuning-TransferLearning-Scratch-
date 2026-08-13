from __future__ import annotations

import torch
from torch import nn

from src.config.experiment import ExperimentConfig
from src.models.applications import build_application_model
from src.models.custom_cnn import CustomCNN
from src.models.qcs import QCSModel
from src.models.poster_v2 import PosterV2


def configure_scenarios_for_custom(model: nn.Module, scenario: str) -> None:
    """
    Configures trainable parameters for custom/SOTA models based on the training scenario.
    """
    backbone = getattr(model, "features", None) or getattr(model, "backbone", None)
    if backbone is None:
        return

    if scenario == "transfer":
        # Freeze the backbone completely
        for param in backbone.parameters():
            param.requires_grad = False
    elif scenario == "fine_tuning":
        # Freeze the backbone, then unfreeze the last block (e.g. layer4 of ResNet or block8 of InceptionResnetV1)
        for param in backbone.parameters():
            param.requires_grad = False
            
        from src.models.qcs import InceptionResnetV1Features
        from src.models.swin_face import SwinTransformer
        if isinstance(backbone, InceptionResnetV1Features):
            # Unfreeze block8 (the last block before global pooling)
            for param in backbone.base.block8.parameters():
                param.requires_grad = True
        elif isinstance(backbone, SwinTransformer):
            # Unfreeze the last basic layer (index 3) and final norm
            for param in backbone.layers[-1].parameters():
                param.requires_grad = True
            if hasattr(backbone, "norm"):
                for param in backbone.norm.parameters():
                    param.requires_grad = True
        elif hasattr(backbone, "blocks") and hasattr(backbone, "norm"):
            # Unfreeze the last block of DeiT / VisionTransformer and final norm
            for param in backbone.blocks[-1].parameters():
                param.requires_grad = True
            for param in backbone.norm.parameters():
                param.requires_grad = True
        else:
            # ResNet models have layer4 as their last block (at index -1 of features sequential list)
            # We can unfreeze it directly
            for child in list(backbone.children())[-1:]:
                for param in child.parameters():
                    param.requires_grad = True
    elif scenario == "scratch":
        # Keep everything trainable
        for param in backbone.parameters():
            param.requires_grad = True


def build_model(config: ExperimentConfig, backbone_name: str | None = None) -> nn.Module:
    if config.architecture == "custom_cnn":
        return CustomCNN(num_classes=config.num_classes, dropout=config.dropout)

    if config.architecture == "qcs":
        # Default to inception_resnet_v1 pre-trained on VGGFace2 for face-specific weights
        bb = backbone_name or "inception_resnet_v1"
        model = QCSModel(
            num_classes=config.num_classes,
            backbone_name=bb,
            dropout=config.dropout
        )
        configure_scenarios_for_custom(model, config.scenario)
        return model

    if config.architecture == "poster_v2":
        # Default to inception_resnet_v1 pre-trained on VGGFace2 for face-specific weights
        bb = backbone_name or "inception_resnet_v1"
        model = PosterV2(
            num_classes=config.num_classes,
            backbone_name=bb,
            dropout=config.dropout
        )
        configure_scenarios_for_custom(model, config.scenario)
        return model

    if config.architecture == "swin_face":
        from src.models.swin_face import SwinFaceFER
        from pathlib import Path
        model = SwinFaceFER(
            num_classes=config.num_classes,
            dropout=config.dropout
        )
        
        # Auto-detect and load face-pretrained weights if they exist in project root
        pretrained_path = Path("swin_face_pretrained.pth")
        if not pretrained_path.exists():
            pretrained_path = Path("swin_face_pretrained.pt")
            
        if pretrained_path.exists():
            print(f"Detectados pesos pre-entrenados para SwinFace: {pretrained_path.name}")
            model.load_pretrained_backbone(str(pretrained_path))
        else:
            print("==============================================================================")
            print("AVISO: No se encontraron pesos pre-entrenados para SwinFace ('swin_face_pretrained.pth'/'pt').")
            print("Necesita cargar pesos pre-entrenados en rostros (normalmente Swin Transformer entrenado en la base de datos de caras WebFace o MS-Celeb-1M) para rendimiento competitivo.")
            print("El entrenamiento se ejecutará con inicialización por defecto.")
            print("==============================================================================")
            
        configure_scenarios_for_custom(model, config.scenario)
        return model

    if config.architecture == "deit":
        from src.models.deit import DeiTFER
        model = DeiTFER(
            num_classes=config.num_classes,
            dropout=config.dropout
        )
        configure_scenarios_for_custom(model, config.scenario)
        return model

    return build_application_model(
        architecture=config.architecture,
        input_shape=config.input_shape,
        num_classes=config.num_classes,
        scenario=config.scenario,
        dropout=config.dropout,
        dense_units=config.dense_units,
        fine_tune_at=config.fine_tune_at,
    )


def build_optimizer(name: str, parameters, learning_rate: float, weight_decay: float = 0.0):
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(parameters, lr=learning_rate, weight_decay=weight_decay)
    return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)


def compile_model(model, config: ExperimentConfig):
    return model

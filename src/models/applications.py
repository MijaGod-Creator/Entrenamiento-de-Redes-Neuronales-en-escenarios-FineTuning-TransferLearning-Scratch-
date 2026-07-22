from __future__ import annotations

import torch
from torch import nn
from torchvision import models


APPLICATIONS = {
    "vgg16": models.vgg16,
    "resnet50": models.resnet50,
    "mobilenetv2": models.mobilenet_v2,
    "efficientnetb0": models.efficientnet_b0,
    "densenet121": models.densenet121,
}


FEATURE_DIMS = {
    "vgg16": 25088,
    "resnet50": 2048,
    "mobilenetv2": 1280,
    "efficientnetb0": 1280,
    "densenet121": 1024,
}


WEIGHTS = {
    "vgg16": models.VGG16_Weights.DEFAULT,
    "resnet50": models.ResNet50_Weights.DEFAULT,
    "mobilenetv2": models.MobileNet_V2_Weights.DEFAULT,
    "efficientnetb0": models.EfficientNet_B0_Weights.DEFAULT,
    "densenet121": models.DenseNet121_Weights.DEFAULT,
}

class TorchEmotionBackbone(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module, architecture: str):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.architecture = architecture

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        if x.ndim > 2:
            x = torch.flatten(x, 1)
        return self.head(x)


def _safe_builder(architecture: str, weights):
    builder = APPLICATIONS[architecture]
    try:
        return builder(weights=weights)
    except Exception:
        return builder(weights=None)


def _set_trainable(module: nn.Module, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def _unfreeze_last_blocks(architecture: str, backbone: nn.Module, fine_tune_at: int) -> None:
    if fine_tune_at == 0:
        return

    if architecture == "resnet50":
        blocks = [backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4]
    elif architecture == "densenet121":
        blocks = [backbone.features.denseblock1, backbone.features.denseblock2, backbone.features.denseblock3, backbone.features.denseblock4]
    elif architecture in {"mobilenetv2", "efficientnetb0", "vgg16"}:
        feature_source = backbone.features if hasattr(backbone, "features") else backbone
        blocks = list(feature_source.children())
    else:
        blocks = list(backbone.children())

    if fine_tune_at < 0:
        count = min(abs(fine_tune_at), len(blocks))
        selected = blocks[-count:]
    else:
        selected = blocks[fine_tune_at:]

    for block in selected:
        _set_trainable(block, True)


def build_application_model(
    architecture: str,
    input_shape: tuple[int, int, int],
    num_classes: int,
    scenario: str,
    dropout: float = 0.35,
    dense_units: int = 256,
    fine_tune_at: int = -30,
) -> nn.Module:
    del input_shape
    architecture = architecture.lower()
    if architecture not in APPLICATIONS:
        raise ValueError(f"Arquitectura no soportada: {architecture}")

    weights = None if scenario == "scratch" else WEIGHTS[architecture]
    backbone = _safe_builder(architecture, weights)

    if architecture == "vgg16":
        backbone.classifier = nn.Identity()
    elif architecture == "resnet50":
        backbone.fc = nn.Identity()
    elif architecture in {"mobilenetv2", "efficientnetb0"}:
        backbone.classifier = nn.Identity()
    elif architecture == "densenet121":
        backbone.classifier = nn.Identity()

    if scenario == "transfer":
        _set_trainable(backbone, False)
    elif scenario == "fine_tuning":
        _set_trainable(backbone, False)
        _unfreeze_last_blocks(architecture, backbone, fine_tune_at)
    elif scenario == "scratch":
        _set_trainable(backbone, True)
    else:
        raise ValueError(f"Escenario no soportado: {scenario}")

    feature_dim = FEATURE_DIMS[architecture]
    head = nn.Sequential(
        nn.Linear(feature_dim, dense_units),
        nn.BatchNorm1d(dense_units),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(dense_units, num_classes),
    )
    return TorchEmotionBackbone(backbone=backbone, head=head, architecture=architecture)

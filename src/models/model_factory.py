from __future__ import annotations

import torch
from torch import nn

from src.config.experiment import ExperimentConfig
from src.models.applications import build_application_model
from src.models.custom_cnn import CustomCNN


def build_model(config: ExperimentConfig) -> nn.Module:
    if config.architecture == "custom_cnn":
        return CustomCNN(num_classes=config.num_classes, dropout=config.dropout)

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

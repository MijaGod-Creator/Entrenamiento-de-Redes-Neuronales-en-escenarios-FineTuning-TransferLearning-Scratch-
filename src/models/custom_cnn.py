from __future__ import annotations

import torch
from torch import nn


class CustomCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.35):
        super().__init__()
        blocks = []
        in_channels = 3
        for filters in (32, 64, 128, 256):
            blocks.extend(
                [
                    nn.Conv2d(in_channels, filters, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(filters),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(filters),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                    nn.Dropout(dropout / 2),
                ]
            )
            in_channels = filters

        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import json

from src.config.settings import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    IMAGE_SIZE,
    NUM_CLASSES,
    RANDOM_SEED,
)


@dataclass
class ExperimentConfig:
    architecture: str = "custom_cnn"
    scenario: str = "scratch"
    image_size: tuple[int, int] = IMAGE_SIZE
    num_classes: int = NUM_CLASSES
    batch_size: int = DEFAULT_BATCH_SIZE
    epochs: int = DEFAULT_EPOCHS
    learning_rate: float = 1e-3
    fine_tuning_learning_rate: float = 1e-5
    optimizer: str = "adam"
    dropout: float = 0.35
    dense_units: int = 256
    weight_decay: float = 0.0
    validation_size: float = 0.15
    use_augmentation: bool = False
    use_mixup: bool = False
    use_cutmix: bool = False
    class_weight: bool = True
    patience: int = 8
    reduce_lr_patience: int = 4
    fine_tune_at: int = -30
    seed: int = RANDOM_SEED

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (*self.image_size, 3)

    @property
    def experiment_name(self) -> str:
        aug = "aug" if self.use_augmentation else "noaug"
        return f"{self.architecture}_{self.scenario}_{aug}"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["image_size"] = list(self.image_size)
        data["input_shape"] = list(self.input_shape)
        data["experiment_name"] = self.experiment_name
        return data

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

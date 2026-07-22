from __future__ import annotations

import itertools
import random
from pathlib import Path

import pandas as pd

from src.config.experiment import ExperimentConfig
from src.config.settings import PROCESSED_DATASET_DIR, TUNING_RESULTS_DIR
from src.preprocessing.data_split import DatasetSplitter
from src.training.trainer import ExperimentTrainer


class HyperparameterTuner:
    def __init__(self, base_config: ExperimentConfig, max_trials: int = 15):
        self.base_config = base_config
        self.max_trials = max_trials
        self.output_dir = TUNING_RESULTS_DIR / base_config.experiment_name

    def run(self):
        if not (PROCESSED_DATASET_DIR / "train.csv").exists():
            DatasetSplitter(output_dir=PROCESSED_DATASET_DIR).run()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        search_space = list(
            itertools.product(
                [16, 32, 64],
                [1e-4, 3e-4, 1e-3],
                ["adam", "adamw", "sgd", "rmsprop"],
                [0.2, 0.3, 0.4, 0.5],
                [128, 256, 512],
                [0.0, 1e-5, 1e-4],
            )
        )
        rng = random.Random(self.base_config.seed)
        rng.shuffle(search_space)
        trials = search_space[: self.max_trials]

        results = []
        for index, (batch_size, learning_rate, optimizer, dropout, dense_units, weight_decay) in enumerate(trials, start=1):
            trial_config = ExperimentConfig(
                architecture=self.base_config.architecture,
                scenario=self.base_config.scenario,
                image_size=self.base_config.image_size,
                batch_size=batch_size,
                epochs=max(3, min(self.base_config.epochs, 8)),
                learning_rate=learning_rate,
                optimizer=optimizer,
                dropout=dropout,
                dense_units=dense_units,
                weight_decay=weight_decay,
                validation_size=self.base_config.validation_size,
                use_augmentation=self.base_config.use_augmentation,
                use_mixup=self.base_config.use_mixup,
                use_cutmix=self.base_config.use_cutmix,
                class_weight=self.base_config.class_weight,
                fine_tune_at=self.base_config.fine_tune_at,
                seed=self.base_config.seed,
            )
            run_name = f"{self.base_config.experiment_name}_trial_{index}"
            result = ExperimentTrainer(trial_config, run_name=run_name).run(epochs_override=trial_config.epochs)
            results.append(
                {
                    "trial": index,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "optimizer": optimizer,
                    "dropout": dropout,
                    "dense_units": dense_units,
                    "weight_decay": weight_decay,
                    "best_val_accuracy": result["best_val_accuracy"],
                    "best_val_loss": result["best_val_loss"],
                }
            )

        results_df = pd.DataFrame(results).sort_values(["best_val_accuracy", "best_val_loss"], ascending=[False, True])
        results_df.to_csv(self.output_dir / "tuning_results.csv", index=False)
        best_row = results_df.iloc[0]
        (self.output_dir / "best_hyperparameters.txt").write_text(
            "\n".join(
                [
                    f"batch_size: {int(best_row['batch_size'])}",
                    f"learning_rate: {best_row['learning_rate']}",
                    f"optimizer: {best_row['optimizer']}",
                    f"dropout: {best_row['dropout']}",
                    f"dense_units: {int(best_row['dense_units'])}",
                    f"weight_decay: {best_row['weight_decay']}",
                    f"best_val_accuracy: {best_row['best_val_accuracy']}",
                    f"best_val_loss: {best_row['best_val_loss']}",
                ]
            ),
            encoding="utf-8",
        )
        return best_row.to_dict()

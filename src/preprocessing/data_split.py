from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.settings import EDA_RESULTS_DIR, PROCESSED_DATASET_DIR, RANDOM_SEED


class DatasetSplitter:
    def __init__(
        self,
        metadata_path: Path = EDA_RESULTS_DIR / "image_metadata.csv",
        output_dir: Path = PROCESSED_DATASET_DIR,
        validation_size: float = 0.15,
        seed: int = RANDOM_SEED,
    ):
        self.metadata_path = metadata_path
        self.output_dir = output_dir
        self.validation_size = validation_size
        self.seed = seed

    def run(self) -> dict[str, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        metadata = pd.read_csv(self.metadata_path)
        metadata = metadata[(~metadata["is_damaged"]) & metadata["label"].notna()].copy()
        metadata["label"] = metadata["label"].astype(int) - 1

        train_full = metadata[metadata["split"] == "train"].copy()
        test_df = metadata[metadata["split"] == "test"].copy()

        train_df, val_df = train_test_split(
            train_full,
            test_size=self.validation_size,
            stratify=train_full["label"],
            random_state=self.seed,
        )

        outputs = {
            "train": self.output_dir / "train.csv",
            "validation": self.output_dir / "validation.csv",
            "test": self.output_dir / "test.csv",
        }
        train_df.to_csv(outputs["train"], index=False)
        val_df.to_csv(outputs["validation"], index=False)
        test_df.to_csv(outputs["test"], index=False)
        return outputs

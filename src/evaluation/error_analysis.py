from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config.settings import EVALUATION_FIGURES_DIR, RAFDB_LABEL_MAP


def save_prediction_examples(
    metadata_csv: Path,
    y_true_path: Path,
    y_prob_path: Path,
    experiment_name: str,
    max_examples: int = 25,
) -> None:
    metadata = pd.read_csv(metadata_csv).reset_index(drop=True)
    y_true = np.load(y_true_path)
    y_prob = np.load(y_prob_path)
    y_pred = np.argmax(y_prob, axis=1)
    metadata = metadata.iloc[: len(y_true)].copy()
    metadata["true"] = y_true
    metadata["pred"] = y_pred
    metadata["confidence"] = np.max(y_prob, axis=1)

    output_dir = EVALUATION_FIGURES_DIR / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_examples(
        metadata[metadata["true"] == metadata["pred"]].sort_values("confidence", ascending=False),
        output_dir / "correct_examples.png",
        "Ejemplos correctos",
        max_examples,
    )
    _plot_examples(
        metadata[metadata["true"] != metadata["pred"]].sort_values("confidence", ascending=False),
        output_dir / "incorrect_examples.png",
        "Errores de clasificacion",
        max_examples,
    )


def _plot_examples(frame: pd.DataFrame, output_path: Path, title: str, max_examples: int) -> None:
    sample = frame.head(max_examples)
    if sample.empty:
        return
    cols = 5
    rows = int(np.ceil(len(sample) / cols))
    plt.figure(figsize=(cols * 2.2, rows * 2.5))
    for idx, (_, row) in enumerate(sample.iterrows()):
        image = plt.imread(row["path"])
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(image)
        plt.axis("off")
        true_name = RAFDB_LABEL_MAP[int(row["true"]) + 1]
        pred_name = RAFDB_LABEL_MAP[int(row["pred"]) + 1]
        plt.title(f"T:{true_name}\nP:{pred_name}", fontsize=8)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

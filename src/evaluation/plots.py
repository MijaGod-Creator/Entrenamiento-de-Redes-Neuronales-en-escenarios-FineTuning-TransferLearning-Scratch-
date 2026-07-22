from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

from src.config.settings import EVALUATION_FIGURES_DIR, NUM_CLASSES, RAFDB_LABEL_MAP, TRAINING_FIGURES_DIR


def plot_history(history_csv: Path, output_dir: Path | None = None) -> None:
    output_dir = output_dir or TRAINING_FIGURES_DIR / history_csv.parent.name
    output_dir.mkdir(parents=True, exist_ok=True)
    history = pd.read_csv(history_csv)

    for metric in ("accuracy", "loss", "top_2_accuracy", "top_3_accuracy"):
        if metric not in history.columns:
            continue
        plt.figure(figsize=(8, 5))
        plt.plot(history[metric], label=f"train_{metric}")
        val_metric = f"val_{metric}"
        if val_metric in history.columns:
            plt.plot(history[val_metric], label=val_metric)
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.title(f"Curva {metric}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}_curve.png", dpi=300)
        plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_prob: np.ndarray, experiment_name: str) -> None:
    output_dir = EVALUATION_FIGURES_DIR / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    y_pred = np.argmax(y_prob, axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[RAFDB_LABEL_MAP[i + 1] for i in range(NUM_CLASSES)],
        yticklabels=[RAFDB_LABEL_MAP[i + 1] for i in range(NUM_CLASSES)],
    )
    plt.xlabel("Prediccion")
    plt.ylabel("Etiqueta real")
    plt.title("Matriz de confusion")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300)
    plt.close()


def plot_roc_curves(y_true: np.ndarray, y_prob: np.ndarray, experiment_name: str) -> None:
    output_dir = EVALUATION_FIGURES_DIR / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true_onehot = np.eye(NUM_CLASSES)[y_true]
    plt.figure(figsize=(9, 7))
    for idx in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, idx], y_prob[:, idx])
        plt.plot(fpr, tpr, label=f"{RAFDB_LABEL_MAP[idx + 1]} AUC={auc(fpr, tpr):.3f}")
    plt.plot([0, 1], [0, 1], "k--", label="Azar")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curvas ROC One-vs-Rest")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves.png", dpi=300)
    plt.close()


def plot_model_comparison(metrics_csv: Path, output_dir: Path = EVALUATION_FIGURES_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(metrics_csv)
    metrics = metrics.sort_values("f1_macro", ascending=False)

    for metric in ("accuracy", "f1_macro", "balanced_accuracy", "inference_ms_per_image", "num_parameters"):
        if metric not in metrics.columns:
            continue
        plt.figure(figsize=(12, 6))
        sns.barplot(data=metrics, x="experiment_name", y=metric)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Comparacion por {metric}")
        plt.tight_layout()
        plt.savefig(output_dir / f"comparison_{metric}.png", dpi=300)
        plt.close()

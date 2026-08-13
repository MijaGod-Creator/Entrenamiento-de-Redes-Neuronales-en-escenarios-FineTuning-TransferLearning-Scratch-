import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from src.config.settings import EVALUATION_RESULTS_DIR, NUM_CLASSES, RAFDB_LABEL_MAP
from src.preprocessing.tfdata import build_dataset


def specificity_per_class(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    values = {}
    for idx in range(num_classes):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        values[str(idx + 1)] = float(tn / (tn + fp)) if (tn + fp) else 0.0
    return values


class ModelEvaluator:
    def __init__(
        self,
        model_path: Path,
        test_csv: Path,
        experiment_name: str,
        batch_size: int = 32,
        image_size: tuple[int, int] = (224, 224),
    ):
        self.model_path = model_path
        self.test_csv = test_csv
        self.experiment_name = experiment_name
        self.batch_size = batch_size
        self.image_size = image_size
        self.output_dir = EVALUATION_RESULTS_DIR / experiment_name

    def run(self) -> dict:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = torch.load(self.model_path, map_location="cpu")
        from src.models.model_factory import build_model
        from src.config.experiment import ExperimentConfig

        config_data = checkpoint["config"]
        allowed_keys = set(ExperimentConfig.__dataclass_fields__.keys())
        config = ExperimentConfig(**{key: value for key, value in config_data.items() if key in allowed_keys})
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Detect backbone automatically from the state_dict to avoid size mismatches
        backbone_override = None
        state_dict = checkpoint["model_state_dict"]
        if config.architecture == "qcs" and "csa.q_proj.weight" in state_dict:
            ch = state_dict["csa.q_proj.weight"].shape[1]
            if ch == 512:
                backbone_override = "resnet18"
            elif ch == 2048:
                backbone_override = "resnet50"
            elif ch == 1792:
                backbone_override = "inception_resnet_v1"
        elif config.architecture == "poster_v2" and "proj.weight" in state_dict:
            ch = state_dict["proj.weight"].shape[1]
            if ch == 512:
                backbone_override = "resnet18"
            elif ch == 2048:
                backbone_override = "resnet50"
            elif ch == 1792:
                backbone_override = "inception_resnet_v1"

        model = build_model(config, backbone_name=backbone_override)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        test_loader = build_dataset(
            self.test_csv,
            batch_size=self.batch_size,
            image_size=self.image_size,
            shuffle=False,
            train=False,
            use_augmentation=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

        y_true_batches = []
        y_prob_batches = []
        start = time.perf_counter()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                logits = model(images)
                probs = F.softmax(logits, dim=1)
                y_prob_batches.append(probs.cpu().numpy())
                y_true_batches.append(labels.cpu().numpy())
        inference_seconds = time.perf_counter() - start

        y_true = np.concatenate(y_true_batches, axis=0)
        y_prob = np.concatenate(y_prob_batches, axis=0)
        y_pred = np.argmax(y_prob, axis=1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )

        metrics = {
            "experiment_name": self.experiment_name,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(f1),
            "precision_weighted": float(weighted_precision),
            "recall_weighted": float(weighted_recall),
            "f1_weighted": float(weighted_f1),
            "specificity_macro": float(np.mean(list(specificity_per_class(y_true, y_pred, NUM_CLASSES).values()))),
            "top_2_accuracy": float(np.mean([true in np.argsort(prob)[-2:] for true, prob in zip(y_true, y_prob)])),
            "top_3_accuracy": float(np.mean([true in np.argsort(prob)[-3:] for true, prob in zip(y_true, y_prob)])),
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
            "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
            "auc_macro_ovr": float(roc_auc_score(np.eye(NUM_CLASSES)[y_true], y_prob, average="macro", multi_class="ovr")),
            "inference_total_seconds": float(inference_seconds),
            "inference_ms_per_image": float((inference_seconds / len(y_true)) * 1000),
            "num_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        }

        pd.DataFrame([metrics]).to_csv(self.output_dir / "metrics.csv", index=False)
        (self.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        pd.DataFrame(confusion_matrix(y_true, y_pred)).to_csv(
            self.output_dir / "confusion_matrix.csv", index=False
        )
        report = classification_report(
            y_true,
            y_pred,
            target_names=[RAFDB_LABEL_MAP[i + 1] for i in range(NUM_CLASSES)],
            output_dict=True,
            zero_division=0,
        )
        pd.DataFrame(report).transpose().to_csv(self.output_dir / "classification_report.csv")
        np.save(self.output_dir / "y_true.npy", y_true)
        np.save(self.output_dir / "y_prob.npy", y_prob)
        self._save_roc_points(np.eye(NUM_CLASSES)[y_true], y_prob)
        return metrics

    def _save_roc_points(self, y_true_onehot: np.ndarray, y_prob: np.ndarray) -> None:
        rows = []
        for idx in range(NUM_CLASSES):
            fpr, tpr, thresholds = roc_curve(y_true_onehot[:, idx], y_prob[:, idx])
            for f, t, thr in zip(fpr, tpr, thresholds):
                rows.append(
                    {
                        "class_id": idx + 1,
                        "class_name": RAFDB_LABEL_MAP[idx + 1],
                        "fpr": f,
                        "tpr": t,
                        "threshold": thr,
                    }
                )
        pd.DataFrame(rows).to_csv(self.output_dir / "roc_points.csv", index=False)

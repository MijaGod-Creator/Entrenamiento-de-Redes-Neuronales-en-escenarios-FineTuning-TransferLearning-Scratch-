from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.utils.class_weight import compute_class_weight

from src.augmentation.augmentations import cutmix, mixup
from src.config.experiment import ExperimentConfig
from src.config.settings import PROCESSED_DATASET_DIR, SAVED_MODELS_DIR, TRAINING_RESULTS_DIR
from src.models.model_factory import build_model, build_optimizer
from src.preprocessing.data_split import DatasetSplitter
from src.preprocessing.tfdata import build_dataset_bundle


def _soft_target_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    self_cure_weights: torch.Tensor | None = None
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    loss = -(targets * log_probs).sum(dim=1)
    if class_weights is not None:
        sample_weights = (targets * class_weights.unsqueeze(0)).sum(dim=1)
        loss = loss * sample_weights
    if self_cure_weights is not None:
        loss = loss * self_cure_weights
    return loss.mean()


def _topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    topk = logits.topk(k, dim=1).indices
    hits = topk.eq(targets.unsqueeze(1)).any(dim=1)
    return float(hits.float().mean().item())


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == targets).float().mean().item())


def compute_self_cure_weights(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes SCN-like confidence weights for each sample in the batch to suppress label noise.
    If the model predicts a different class with high confidence (> 0.85) but the ground truth
    target class has very low probability (< 0.15), it assumes a noisy label and dampens the loss by 0.15.
    """
    with torch.no_grad():
        probs = F.softmax(logits, dim=1)
        B = logits.size(0)
        target_probs = probs[range(B), targets]
        max_probs, preds = probs.max(dim=1)
        
        weights = torch.ones(B, device=logits.device)
        # Mislabeled criteria
        mislabeled_mask = (preds != targets) & (max_probs > 0.85) & (target_probs < 0.15)
        weights[mislabeled_mask] = 0.15
        return weights


class ExperimentTrainer:
    def __init__(
        self,
        config: ExperimentConfig,
        processed_dir: Path = PROCESSED_DATASET_DIR,
        resume: bool = False,
        run_name: str | None = None,
    ):
        self.config = config
        self.processed_dir = processed_dir
        self.resume = resume
        self.run_name = run_name or config.experiment_name
        self.result_dir = TRAINING_RESULTS_DIR / self.run_name
        self.model_dir = SAVED_MODELS_DIR / self.run_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare_data(self) -> dict:
        if not (self.processed_dir / "train.csv").exists():
            DatasetSplitter(output_dir=self.processed_dir, validation_size=self.config.validation_size).run()

        loaders = build_dataset_bundle(
            self.processed_dir,
            batch_size=self.config.batch_size,
            image_size=self.config.image_size,
            use_augmentation=self.config.use_augmentation,
            seed=self.config.seed,
            num_workers=self._num_workers(),
            pin_memory=self.device.type == "cuda",
        )
        return loaders

    def compute_class_weights(self) -> dict[int, float] | None:
        if not self.config.class_weight:
            return None
        labels = pd.read_csv(self.processed_dir / "train.csv")["label"].astype(int).to_numpy()
        classes = np.unique(labels)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
        return {int(cls): float(weight) for cls, weight in zip(classes, weights)}

    def run(self, epochs_override: int | None = None) -> dict:
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.config.save(self.result_dir / "config.json")

        loaders = self.prepare_data()
        model = build_model(self.config).to(self.device)
        learning_rate = self.config.fine_tuning_learning_rate if self.config.scenario == "fine_tuning" else self.config.learning_rate
        optimizer = build_optimizer(self.config.optimizer, model.parameters(), learning_rate, self.config.weight_decay)
        
        # Use Cosine Annealing Scheduler for long deep runs (epochs >= 45) to ensure smooth SOTA convergence
        if self.config.epochs >= 45:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.epochs,
                eta_min=1e-6
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.3,
                patience=self.config.reduce_lr_patience,
                min_lr=1e-7,
            )
        scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")
        class_weight = self.compute_class_weights()
        class_weight_tensor = None
        if class_weight is not None:
            class_weight_tensor = torch.tensor(
                [class_weight[idx] for idx in sorted(class_weight)],
                dtype=torch.float32,
                device=self.device,
            )

        checkpoint_path = self.model_dir / "latest_model.pt"
        best_path = self.model_dir / "best_model.pt"
        final_path = self.model_dir / "final_model.pt"
        history_path = self.result_dir / "training_log.csv"
        start_epoch = 0
        best_val_loss = float("inf")
        best_val_accuracy = 0.0
        history_rows: list[dict] = []

        if self.resume:
            resume_candidates = [checkpoint_path, best_path]
            loaded_path = None
            last_error: Exception | None = None

            for candidate_path in resume_candidates:
                if not candidate_path.exists():
                    continue
                try:
                    checkpoint = torch.load(candidate_path, map_location=self.device)
                    model.load_state_dict(checkpoint["model_state_dict"])
                    start_epoch = int(checkpoint["epoch"]) + 1
                    best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
                    best_val_accuracy = float(checkpoint.get("best_val_accuracy", best_val_accuracy))
                    if "optimizer_state_dict" in checkpoint:
                        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    if "scheduler_state_dict" in checkpoint:
                        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                    if "scaler_state_dict" in checkpoint:
                        scaler.load_state_dict(checkpoint["scaler_state_dict"])
                    loaded_path = candidate_path
                    break
                except Exception as exc:
                    last_error = exc
                    print(f"No se pudo cargar {candidate_path.name}: {exc}")

            if loaded_path is not None:
                if history_path.exists():
                    history_rows = [
                        row
                        for row in pd.read_csv(history_path).to_dict(orient="records")
                        if int(row.get("epoch", 0)) <= start_epoch
                    ]
                print(f"Reanudando desde: {loaded_path} | epoca inicial: {start_epoch}")
            elif last_error is not None:
                print("No se pudo reanudar desde ningun checkpoint; se iniciara desde cero.")

        total_epochs = epochs_override or self.config.epochs
        start = time.perf_counter()
        patience_counter = 0

        for epoch in range(start_epoch, total_epochs):
            train_metrics = self._run_epoch(
                model=model,
                loader=loaders["train"],
                optimizer=optimizer,
                scaler=scaler,
                class_weight_tensor=class_weight_tensor,
                train=True,
            )
            val_metrics = self._run_epoch(
                model=model,
                loader=loaders["validation"],
                optimizer=None,
                scaler=scaler,
                class_weight_tensor=class_weight_tensor,
                train=False,
            )

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

            row = {
                "epoch": epoch + 1,
                "loss": train_metrics["loss"],
                "accuracy": train_metrics["accuracy"],
                "top_2_accuracy": train_metrics["top_2_accuracy"],
                "top_3_accuracy": train_metrics["top_3_accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_top_2_accuracy": val_metrics["top_2_accuracy"],
                "val_top_3_accuracy": val_metrics["top_3_accuracy"],
                "lr": optimizer.param_groups[0]["lr"],
            }
            history_rows.append(row)
            pd.DataFrame(history_rows).to_csv(history_path, index=False)

            if val_metrics["accuracy"] > best_val_accuracy + 1e-5:
                best_val_loss = val_metrics["loss"]
                best_val_accuracy = val_metrics["accuracy"]
                self._save_checkpoint(
                    best_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    best_val_loss=best_val_loss,
                    best_val_accuracy=best_val_accuracy,
                )
                patience_counter = 0
            else:
                patience_counter += 1

            self._save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_val_loss=best_val_loss,
                best_val_accuracy=best_val_accuracy,
            )

            if patience_counter >= self.config.patience:
                print(f"Early stopping en epoca {epoch + 1}")
                break

        elapsed = time.perf_counter() - start
        history_df = pd.DataFrame(history_rows)
        history_df.to_csv(self.result_dir / "history.csv", index=False)
        history_df.to_csv(self.result_dir / "training_log.csv", index=False)
        self._save_checkpoint(
            final_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=int(history_df["epoch"].iloc[-1]) - 1 if not history_df.empty else 0,
            best_val_loss=best_val_loss,
            best_val_accuracy=best_val_accuracy,
        )
        (self.result_dir / "training_time.json").write_text(
            json.dumps({"seconds": elapsed, "minutes": elapsed / 60}, indent=2),
            encoding="utf-8",
        )
        return {
            "model": model,
            "history": history_df,
            "best_val_loss": best_val_loss,
            "best_val_accuracy": best_val_accuracy,
            "checkpoint_path": str(checkpoint_path),
            "best_path": str(best_path),
            "final_path": str(final_path),
        }

    def _num_workers(self) -> int:
        if os.name == "nt":
            return 0
        cpu_count = os.cpu_count() or 0
        return 2 if cpu_count >= 4 else 0

    def _run_epoch(
        self,
        model: nn.Module,
        loader,
        optimizer,
        scaler,
        class_weight_tensor: torch.Tensor | None,
        train: bool,
    ) -> dict[str, float]:
        model.train(train)
        total_loss = 0.0
        total_samples = 0
        all_targets = []
        all_logits = []

        for images, targets in loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            original_targets = targets
            soft_targets = F.one_hot(targets, num_classes=self.config.num_classes).float()

            if train and self.config.use_mixup and self.config.use_cutmix:
                if torch.rand(1).item() < 0.5:
                    images, soft_targets = mixup(images, soft_targets)
                else:
                    images, soft_targets = cutmix(images, soft_targets)
            elif train and self.config.use_mixup:
                images, soft_targets = mixup(images, soft_targets)
            elif train and self.config.use_cutmix:
                images, soft_targets = cutmix(images, soft_targets)

            with torch.set_grad_enabled(train):
                with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                    logits = model(images)
                    sc_weights = None
                    if train and self.config.self_cure:
                        sc_weights = compute_self_cure_weights(logits, original_targets)
                    loss = _soft_target_cross_entropy(logits, soft_targets, class_weight_tensor, sc_weights)

                if train and optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

            batch_size = images.size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            all_targets.append(original_targets.detach().cpu())
            all_logits.append(logits.detach().cpu())

        targets_tensor = torch.cat(all_targets)
        logits_tensor = torch.cat(all_logits)
        return {
            "loss": total_loss / max(total_samples, 1),
            "accuracy": _accuracy(logits_tensor, targets_tensor),
            "top_2_accuracy": _topk_accuracy(logits_tensor, targets_tensor, 2),
            "top_3_accuracy": _topk_accuracy(logits_tensor, targets_tensor, 3),
        }

    def _save_checkpoint(
        self,
        path: Path,
        model: nn.Module,
        optimizer,
        scheduler,
        scaler,
        epoch: int,
        best_val_loss: float,
        best_val_accuracy: float,
    ) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_val_loss": best_val_loss,
                "best_val_accuracy": best_val_accuracy,
                "config": self.config.to_dict(),
                "run_name": self.run_name,
            },
            path,
            _use_new_zipfile_serialization=False,
        )


def run_experiment_grid(
    architectures: list[str],
    scenarios: list[str],
    use_augmentation_values: list[bool],
    base_config: ExperimentConfig,
) -> None:
    for architecture in architectures:
        for scenario in scenarios:
            for use_augmentation in use_augmentation_values:
                config = replace(
                    base_config,
                    architecture=architecture,
                    scenario=scenario,
                    use_augmentation=use_augmentation,
                )
                trainer = ExperimentTrainer(config)
                training_time_path = trainer.result_dir / "training_time.json"
                latest_checkpoint_path = trainer.model_dir / "latest_model.pt"

                if training_time_path.exists():
                    print(f"Saltando {config.experiment_name}: ya estaba completado.")
                    continue

                resume = latest_checkpoint_path.exists()
                if resume:
                    print(f"Reanudando {config.experiment_name} desde checkpoint parcial.")
                trainer.resume = resume
                trainer.run()

from pathlib import Path

from src.config.settings import SAVED_MODELS_DIR, TRAINING_RESULTS_DIR


def latest_model_path(experiment_name: str) -> Path:
    return SAVED_MODELS_DIR / experiment_name / "latest_model.pt"


def best_model_path(experiment_name: str) -> Path:
    return SAVED_MODELS_DIR / experiment_name / "best_model.pt"


def final_model_path(experiment_name: str) -> Path:
    return SAVED_MODELS_DIR / experiment_name / "final_model.pt"


def training_log_path(experiment_name: str) -> Path:
    return TRAINING_RESULTS_DIR / experiment_name / "training_log.csv"

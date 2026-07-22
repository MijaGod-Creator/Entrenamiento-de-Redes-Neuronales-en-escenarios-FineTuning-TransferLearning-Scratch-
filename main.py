import argparse
import json
from pathlib import Path

from src.config.settings import (
    DATASET_DIR,
    EDA_FIGURES_DIR,
    EDA_RESULTS_DIR,
    FIGURES_DIR,
    LOGS_DIR,
    PROJECT_ROOT,
    RAW_DATASET_DIR,
    RESULTS_DIR,
    SAVED_MODELS_DIR,
    PROCESSED_DATASET_DIR,
    SUPPORTED_ARCHITECTURES,
    SUPPORTED_SCENARIOS,
    TRAINING_RESULTS_DIR,
    EVALUATION_RESULTS_DIR,
    TUNING_RESULTS_DIR,
    ARTICLE_RESULTS_DIR,
    ZIP_CANDIDATE_NAMES,
)
from src.config.experiment import ExperimentConfig
from src.evaluation.comparison import aggregate_metrics
from src.evaluation.eda import EDAReport
from src.evaluation.metrics import ModelEvaluator
from src.evaluation.plots import plot_model_comparison
from src.preprocessing.dataset_inspector import RAFDBInspector
from src.preprocessing.data_split import DatasetSplitter
from src.training.trainer import ExperimentTrainer, run_experiment_grid
from src.utils.files import ensure_directories, find_dataset_zip
from src.utils.reporting import generate_ieee_article
from src.utils.reproducibility import set_global_seed
from src.utils.system import configure_torch_runtime, save_environment_report


def run_eda() -> None:
    ensure_directories(
        [
            DATASET_DIR,
            RAW_DATASET_DIR,
            RESULTS_DIR,
            FIGURES_DIR,
            LOGS_DIR,
            SAVED_MODELS_DIR,
            PROCESSED_DATASET_DIR,
            EDA_RESULTS_DIR,
            EDA_FIGURES_DIR,
            TRAINING_RESULTS_DIR,
            EVALUATION_RESULTS_DIR,
            TUNING_RESULTS_DIR,
            ARTICLE_RESULTS_DIR,
        ]
    )

    zip_path = find_dataset_zip(PROJECT_ROOT, ZIP_CANDIDATE_NAMES)
    inspector = RAFDBInspector(zip_path=zip_path, extraction_dir=RAW_DATASET_DIR / "rafdb")

    zip_index = inspector.inspect_zip()
    inspector.extract_if_needed()
    structure = inspector.detect_structure()
    metadata, label_table = inspector.build_metadata(structure)

    detected_structure = {
        "zip_path": str(structure.zip_path),
        "extraction_dir": str(structure.extraction_dir),
        "dataset_root": str(structure.dataset_root),
        "label_files": [str(path) for path in structure.label_files],
        "split_dirs": {name: str(path) for name, path in structure.split_dirs.items()},
        "num_image_files": len(structure.image_files),
    }
    (EDA_RESULTS_DIR / "detected_structure.json").write_text(
        json.dumps(detected_structure, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report = EDAReport(metadata=metadata, zip_index=zip_index, label_table=label_table)
    summary = report.save_all()

    print("ETAPA 1 completada.")
    print(f"ZIP utilizado: {zip_path}")
    print(f"Dataset extraido en: {structure.extraction_dir}")
    print(f"Raiz detectada: {structure.dataset_root}")
    print(f"Archivos de etiquetas: {[str(p) for p in structure.label_files]}")
    print(f"Splits detectados: {structure.split_dirs}")
    print(f"Total imagenes: {summary['total_images']}")
    print(f"Train: {summary['train_images']} | Test: {summary['test_images']}")
    print(f"Imagenes danadas: {summary['damaged_images']}")
    print(f"Etiquetas nulas: {summary['null_labels']}")
    print(f"Duplicados: {summary['duplicate_images']}")
    print(f"Desbalance max/min: {summary['imbalance_ratio_max_min']}")
    print(f"Resultados CSV/JSON: {EDA_RESULTS_DIR}")
    print(f"Figuras PNG: {EDA_FIGURES_DIR}")


def run_preprocess(validation_size: float) -> None:
    outputs = DatasetSplitter(validation_size=validation_size).run()
    print("ETAPA 2 completada.")
    for split, path in outputs.items():
        print(f"{split}: {path}")


def build_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        architecture=args.architecture,
        scenario=args.scenario,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        dropout=args.dropout,
        dense_units=args.dense_units,
        weight_decay=args.weight_decay,
        validation_size=args.validation_size,
        use_augmentation=args.augmentation,
        use_mixup=args.mixup,
        use_cutmix=args.cutmix,
        class_weight=not args.no_class_weight,
        fine_tune_at=args.fine_tune_at,
    )


def run_train(args: argparse.Namespace) -> None:
    config = build_config_from_args(args)
    ExperimentTrainer(config, resume=args.resume).run()
    print(f"Entrenamiento completado: {config.experiment_name}")


def run_train_grid(args: argparse.Namespace) -> None:
    base_config = build_config_from_args(args)
    architectures = list(SUPPORTED_ARCHITECTURES) if args.architecture == "all" else [args.architecture]
    scenarios = list(SUPPORTED_SCENARIOS) if args.scenario == "all" else [args.scenario]
    run_experiment_grid(architectures, scenarios, [False, True], base_config)
    print("Grid de experimentos completado.")


def run_tuning(args: argparse.Namespace) -> None:
    try:
        from src.training.tuner import HyperparameterTuner
    except ModuleNotFoundError as exc:
        if exc.name == "keras_tuner":
            raise ModuleNotFoundError(
                "Falta keras-tuner. Instale la dependencia con: pip install keras-tuner"
            ) from exc
        raise

    config = build_config_from_args(args)
    best_hp = HyperparameterTuner(config, max_trials=args.max_trials).run()
    print("Mejores hiperparametros:")
    for name in best_hp.values.keys():
        print(f"{name}: {best_hp.get(name)}")


def run_evaluate(args: argparse.Namespace) -> None:
    config = build_config_from_args(args)
    model_path = Path(args.model_path) if args.model_path else SAVED_MODELS_DIR / config.experiment_name / "best_model.keras"
    evaluator = ModelEvaluator(
        model_path=model_path,
        test_csv=PROCESSED_DATASET_DIR / "test.csv",
        experiment_name=config.experiment_name,
        batch_size=args.batch_size,
        image_size=config.image_size,
    )
    metrics = evaluator.run()
    print(json.dumps(metrics, indent=2))


def run_compare() -> None:
    comparison = aggregate_metrics()
    plot_model_comparison(PROJECT_ROOT / "results" / "model_comparison.csv")
    print(comparison.to_string(index=False))


def run_article() -> None:
    path = generate_ieee_article()
    print(f"Articulo generado: {path}")


def run_project_setup(args: argparse.Namespace) -> None:
    run_eda()
    run_preprocess(args.validation_size)
    save_environment_report(PROJECT_ROOT / "results" / "environment.json")
    run_article()
    print("Preparacion completa. Los entrenamientos largos se ejecutan con --stage train o --stage train-grid.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Proyecto RAF-DB CNN")
    parser.add_argument(
        "--stage",
        choices=["eda", "preprocess", "train", "train-grid", "tune", "evaluate", "compare", "article", "setup"],
        default="setup",
        help="Etapa a ejecutar.",
    )
    parser.add_argument("--architecture", default="custom_cnn", choices=[*SUPPORTED_ARCHITECTURES, "all"])
    parser.add_argument("--scenario", default="scratch", choices=[*SUPPORTED_SCENARIOS, "all"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", default="adam", choices=["adam", "adamw", "sgd", "rmsprop"])
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--dense-units", type=int, default=256)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--augmentation", action="store_true")
    parser.add_argument("--mixup", action="store_true")
    parser.add_argument("--cutmix", action="store_true")
    parser.add_argument("--no-class-weight", action="store_true")
    parser.add_argument("--fine-tune-at", type=int, default=-30)
    parser.add_argument("--max-trials", type=int, default=15)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    set_global_seed()
    args = parse_args()
    if args.stage == "eda":
        run_eda()
    elif args.stage == "preprocess":
        run_preprocess(args.validation_size)
    elif args.stage == "train":
        configure_torch_runtime()
        set_global_seed(include_torch=True)
        run_train(args)
    elif args.stage == "train-grid":
        configure_torch_runtime()
        set_global_seed(include_torch=True)
        run_train_grid(args)
    elif args.stage == "tune":
        configure_torch_runtime()
        set_global_seed(include_torch=True)
        run_tuning(args)
    elif args.stage == "evaluate":
        configure_torch_runtime()
        set_global_seed(include_torch=True)
        run_evaluate(args)
    elif args.stage == "compare":
        run_compare()
    elif args.stage == "article":
        run_article()
    elif args.stage == "setup":
        run_project_setup(args)

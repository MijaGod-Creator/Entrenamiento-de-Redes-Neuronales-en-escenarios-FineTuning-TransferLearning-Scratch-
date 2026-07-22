from pathlib import Path

import pandas as pd

from src.config.settings import EVALUATION_RESULTS_DIR, RESULTS_DIR


def aggregate_metrics(output_path: Path = RESULTS_DIR / "model_comparison.csv") -> pd.DataFrame:
    rows = []
    for metrics_file in EVALUATION_RESULTS_DIR.glob("*/metrics.csv"):
        frame = pd.read_csv(metrics_file)
        rows.append(frame)
    if not rows:
        raise FileNotFoundError("No hay metricas evaluadas en results/evaluation/*/metrics.csv")
    comparison = pd.concat(rows, ignore_index=True)
    comparison.to_csv(output_path, index=False)
    try:
        comparison.to_excel(output_path.with_suffix(".xlsx"), index=False)
    except ImportError:
        pass
    return comparison

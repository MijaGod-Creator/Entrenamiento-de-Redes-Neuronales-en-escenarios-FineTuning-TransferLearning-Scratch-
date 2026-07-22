import json
import platform
from pathlib import Path

import torch


def configure_torch_runtime() -> dict:
    cuda_available = torch.cuda.is_available()
    gpu_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())] if cuda_available else []

    if cuda_available:
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        print(f"GPU detectada: {gpu_names}")
    else:
        print(
            "Aviso: PyTorch no detecta GPU/CUDA en este entorno. "
            "Para GPU nativa en Windows instala el build CUDA de PyTorch y verifica que el driver soporte la version elegida."
        )

    return {"gpus": gpu_names, "cuda_enabled": cuda_available}


def save_environment_report(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpus": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
    }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

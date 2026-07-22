import os
import random

import numpy as np


def set_global_seed(seed: int = 42, include_torch: bool = False) -> None:
    """Set deterministic seeds for the libraries used in the project."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if not include_torch:
        return

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(False)
            except Exception:
                pass
    except Exception:
        # PyTorch is not required for the EDA-only execution path.
        pass

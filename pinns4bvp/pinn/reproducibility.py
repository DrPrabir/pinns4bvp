"""Reproducibility helpers for PINN training."""

from __future__ import annotations

import os
import random

import numpy as np


def set_reproducibility(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch and request deterministic algorithms.

    Determinism can depend on the selected PyTorch device and operation.  CPU is
    the default device for the alpha releases because it is the easiest target
    for repeatable BVP training.
    """

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

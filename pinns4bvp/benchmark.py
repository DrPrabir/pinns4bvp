"""Backend-independent solution comparison utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ComparisonMetrics:
    rmse: float
    mae: float
    max_abs_error: float
    relative_l2: float


def compare_solutions(reference, candidate, *, variable=0, x=None) -> ComparisonMetrics:
    """Compare two BVP solutions on a common one-dimensional grid."""

    if x is None:
        a = max(reference.problem.a, candidate.problem.a)
        b = min(reference.problem.b, candidate.problem.b)
        x = np.linspace(a, b, 501)
    x = np.asarray(x, dtype=float)
    ref = np.asarray(reference.values(variable, x), dtype=float)
    cand = np.asarray(candidate.values(variable, x), dtype=float)
    error = cand - ref
    denom = np.linalg.norm(ref.ravel())
    relative = np.linalg.norm(error.ravel()) / denom if denom > 0 else np.inf
    return ComparisonMetrics(
        rmse=float(np.sqrt(np.mean(error**2))),
        mae=float(np.mean(np.abs(error))),
        max_abs_error=float(np.max(np.abs(error))),
        relative_l2=float(relative),
    )

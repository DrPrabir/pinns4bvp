"""Error metrics for BVP benchmarking."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ComparisonMetrics:
    """Standard pointwise/global error metrics on a common grid."""

    rmse: float
    mae: float
    max_abs_error: float
    relative_l2: float
    l2_error: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ParameterMetrics:
    """Error metrics for one scalar estimated parameter."""

    reference: float
    estimate: float
    abs_error: float
    relative_error: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def error_metrics(reference, candidate) -> ComparisonMetrics:
    """Compute error metrics for two equally shaped arrays."""

    ref = np.asarray(reference, dtype=float)
    cand = np.asarray(candidate, dtype=float)
    if ref.shape != cand.shape:
        raise ValueError(
            "reference and candidate must have the same shape; "
            f"got {ref.shape} and {cand.shape}"
        )
    if ref.size == 0:
        raise ValueError("reference and candidate must not be empty")
    if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(cand)):
        raise ValueError("reference and candidate must contain only finite values")

    error = cand - ref
    l2_error = float(np.linalg.norm(error.ravel()))
    denom = float(np.linalg.norm(ref.ravel()))
    if denom > 0.0:
        relative_l2 = l2_error / denom
    else:
        relative_l2 = 0.0 if l2_error == 0.0 else float("inf")

    return ComparisonMetrics(
        rmse=float(np.sqrt(np.mean(error**2))),
        mae=float(np.mean(np.abs(error))),
        max_abs_error=float(np.max(np.abs(error))),
        relative_l2=float(relative_l2),
        l2_error=l2_error,
    )


def parameter_metrics(reference: float, estimate: float) -> ParameterMetrics:
    """Compare one scalar estimated parameter with a reference value."""

    reference = float(reference)
    estimate = float(estimate)
    if not np.isfinite(reference) or not np.isfinite(estimate):
        raise ValueError("parameter values must be finite")
    abs_error = abs(estimate - reference)
    relative_error = abs_error / abs(reference) if reference != 0.0 else (
        0.0 if abs_error == 0.0 else float("inf")
    )
    return ParameterMetrics(
        reference=reference,
        estimate=estimate,
        abs_error=float(abs_error),
        relative_error=float(relative_error),
    )


def compare_solutions(reference, candidate, *, variable=0, x=None) -> ComparisonMetrics:
    """Compare two :class:`BVPSolution` objects on a common grid."""

    if x is None:
        a = max(reference.problem.a, candidate.problem.a)
        b = min(reference.problem.b, candidate.problem.b)
        if not a < b:
            raise ValueError("solutions do not have an overlapping domain")
        x = np.linspace(a, b, 501)
    x = np.asarray(x, dtype=float)
    ref = reference.values(variable, x)
    cand = candidate.values(variable, x)
    return error_metrics(ref, cand)

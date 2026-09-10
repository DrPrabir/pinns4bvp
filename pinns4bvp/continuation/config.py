"""Configuration objects for natural parameter continuation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ContinuationConfig:
    """Control continuation stepping and failure recovery.

    Parameters
    ----------
    adaptive:
        If ``True``, failed steps are retried through smaller intermediate
        parameter increments.
    min_step:
        Smallest absolute parameter increment used for adaptive recovery.
    max_step:
        Optional largest absolute increment between consecutive accepted
        solutions. Larger requested gaps are subdivided proactively.
    reduction_factor:
        Fraction of a failed interval used to place a recovery point. Must lie
        strictly between 0 and 1. The default 0.5 inserts midpoints.
    max_retries:
        Maximum recursive recovery depth for a requested target.
    stop_on_failure:
        Stop the full continuation after a requested value cannot be reached.
    reuse_unknown_parameters:
        Initialize unknown BVP parameters from the previous accepted solution.
    pinn_warm_start:
        For the PINN backend, initialize a new network from the previous
        accepted PINN model when compatible.
    record_residuals:
        Evaluate an independent residual report for accepted solutions.
    residual_points:
        Number of points used for continuation residual diagnostics.
    """

    adaptive: bool = True
    min_step: float = 1e-3
    max_step: float | None = None
    reduction_factor: float = 0.5
    max_retries: int = 8
    stop_on_failure: bool = False
    reuse_unknown_parameters: bool = True
    pinn_warm_start: bool = True
    record_residuals: bool = True
    residual_points: int = 201

    def __post_init__(self) -> None:
        if self.min_step <= 0:
            raise ValueError("min_step must be positive")
        if self.max_step is not None and self.max_step <= 0:
            raise ValueError("max_step must be positive or None")
        if self.max_step is not None and self.max_step < self.min_step:
            raise ValueError("max_step must be >= min_step")
        if not 0.0 < self.reduction_factor < 1.0:
            raise ValueError("reduction_factor must lie strictly between 0 and 1")
        if not isinstance(self.max_retries, int) or self.max_retries < 0:
            raise ValueError("max_retries must be an integer >= 0")
        if not isinstance(self.residual_points, int) or self.residual_points < 2:
            raise ValueError("residual_points must be an integer >= 2")

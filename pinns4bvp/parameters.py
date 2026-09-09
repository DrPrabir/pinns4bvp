"""Unknown-parameter definitions for PINNs4BVP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class UnknownParameter:
    """Describe a scalar parameter that must be solved with the BVP state.

    Parameters
    ----------
    initial:
        Initial value supplied to the numerical or PINN backend.
    description:
        Optional human-readable description used in reports.

    Notes
    -----
    v0.5 deliberately supports unconstrained scalar unknown parameters. Bounds
    and transformed/constrained parameters are reserved for a later release.
    """

    initial: float
    description: str | None = None

    def __post_init__(self) -> None:
        value = float(self.initial)
        if not np.isfinite(value):
            raise ValueError("UnknownParameter.initial must be finite")
        object.__setattr__(self, "initial", value)
        if self.description is not None and not isinstance(self.description, str):
            raise TypeError("UnknownParameter.description must be a string or None")


# Concise public alias.
Unknown = UnknownParameter


__all__ = ["UnknownParameter", "Unknown"]

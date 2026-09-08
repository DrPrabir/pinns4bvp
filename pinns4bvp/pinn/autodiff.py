"""Automatic-differentiation helpers for PINN state derivatives."""

from __future__ import annotations


def state_derivative(*args, **kwargs):
    """Return dy/dx for all predicted state variables.

    TODO(v0.2): implement using ``torch.autograd.grad`` while preserving the
    computation graph for optimization.
    """
    raise NotImplementedError(
        "PINN automatic differentiation is not implemented in the v0.2 scaffold yet."
    )

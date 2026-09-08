"""Automatic-differentiation helpers for PINN state derivatives."""

from __future__ import annotations


def derivative(y, x, *, order: int = 1):
    """Differentiate a scalar network output with respect to ``x``.

    ``y`` should have shape ``(N,)`` or ``(N, 1)`` and ``x`` should have shape
    ``(N, 1)``.  Higher orders are computed recursively while preserving the
    computation graph.
    """

    import torch

    if order < 0:
        raise ValueError("order must be >= 0")
    result = y
    if order == 0:
        return result
    for _ in range(order):
        result = torch.autograd.grad(
            result,
            x,
            grad_outputs=torch.ones_like(result),
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0]
    return result


def state_derivative(y, x):
    """Return ``dy/dx`` for all state variables.

    Parameters
    ----------
    y:
        Tensor with shape ``(N, n_equations)``.
    x:
        Tensor with shape ``(N, 1)`` and ``requires_grad=True``.
    """

    import torch

    if y.ndim != 2:
        raise ValueError("y must have shape (N, n_equations)")
    derivatives = [derivative(y[:, i : i + 1], x) for i in range(y.shape[1])]
    return torch.cat(derivatives, dim=1)

"""PyTorch PINN backend for boundary-value problems.

This module is a v0.2 development scaffold.  The public function is present so
we can design a stable backend contract before implementing optimization.
"""

from __future__ import annotations


def solve_with_pinn(*args, **kwargs):
    """Solve a BVP with a physics-informed neural network.

    TODO(v0.2): add collocation sampling, ODE residual loss, boundary loss,
    Adam optimization, optional L-BFGS refinement, and solution evaluation.
    """
    raise NotImplementedError(
        "The PyTorch PINN backend is not implemented in the v0.2 scaffold yet."
    )

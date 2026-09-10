import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pinns4bvp import BVPProblem, solve
from pinns4bvp.pinn import PINNConfig


def _np_eq(x, y, p):
    return np.vstack((y[1], -np.ones_like(x)))


def _np_bc(ya, yb, p):
    return np.array((ya[0], yb[0]))


def _torch_eq(x, y, p):
    return torch.stack((y[1], -torch.ones_like(x)))


def _torch_bc(ya, yb, p):
    return torch.stack((ya[0], yb[0]))


def test_pinn_solution_can_warm_start_another_pinn_solve():
    problem = BVPProblem(
        _np_eq,
        _np_bc,
        (0.0, 1.0),
        2,
        variable_names=("y", "yp"),
        pinn_equations=_torch_eq,
        pinn_boundary_conditions=_torch_bc,
    )
    config = PINNConfig(
        hidden_layers=(8, 8),
        n_collocation=24,
        adam_epochs=20,
        adam_lr=1e-3,
        use_lbfgs=False,
        loss_tolerance=1.0,
        early_stopping_patience=None,
        verbose=False,
    )
    first = solve(problem, method="pinn", pinn_config=config)
    second = solve(
        problem,
        method="pinn",
        pinn_config=config,
        pinn_warm_start=first,
    )
    assert second.metadata["warm_start_used"] is True
    assert second.metadata["model"] is not first.metadata["model"]

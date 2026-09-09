import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pinns4bvp import BVPProblem, UnknownParameter, solve
from pinns4bvp.pinn import PINNConfig


def _np_eq(x, y, p):
    k = p["k"]
    return np.vstack((y[1], -(k**2) * y[0]))


def _np_bc(ya, yb, p):
    return np.array((ya[0], yb[0], ya[1] - 1.0))


def _torch_eq(x, y, p):
    k = p["k"]
    return torch.stack((y[1], -(k**2) * y[0]))


def _torch_bc(ya, yb, p):
    return torch.stack((ya[0], yb[0], ya[1] - 1.0))


def test_pinn_learns_unknown_eigen_parameter():
    problem = BVPProblem(
        _np_eq,
        _np_bc,
        (0.0, 1.0),
        2,
        unknown_parameters={"k": UnknownParameter(3.0)},
        variable_names=("y", "yp"),
        pinn_equations=_torch_eq,
        pinn_boundary_conditions=_torch_bc,
    )
    config = PINNConfig(
        hidden_layers=(24, 24),
        n_collocation=72,
        adam_epochs=1200,
        adam_lr=1.5e-3,
        use_lbfgs=True,
        lbfgs_max_iter=120,
        bc_weight=20.0,
        seed=123,
        history_every=100,
        verbose=False,
        early_stopping_patience=None,
    )
    sol = solve(problem, method="pinn", pinn_config=config)
    assert abs(sol.parameters["k"] - np.pi) < 2e-2
    history = sol.metadata["training_history"]
    assert history.parameter_history("k")
    assert abs(history.final.parameters["k"] - sol.parameters["k"]) < 1e-12

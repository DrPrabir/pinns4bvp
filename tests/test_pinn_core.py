import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pinns4bvp import BVPProblem, solve
from pinns4bvp.pinn import PINNConfig
from pinns4bvp.pinn.autodiff import derivative, state_derivative
from pinns4bvp.pinn.network import build_network
from pinns4bvp.pinn.reproducibility import set_reproducibility


def test_network_shape_and_domain_normalization():
    set_reproducibility(1)
    model = build_network(3, (2.0, 4.0), hidden_layers=(8, 8))
    x = torch.tensor([[2.0], [3.0], [4.0]], dtype=torch.float64)
    model = model.double()
    y = model(x)
    assert y.shape == (3, 3)
    z = model.normalize(x)
    assert torch.allclose(z[:, 0], torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64))


def test_autodiff_scalar_and_state():
    x = torch.linspace(-1.0, 1.0, 11, dtype=torch.float64)[:, None]
    x.requires_grad_(True)
    y1 = x**2
    y2 = x**3
    assert torch.allclose(derivative(y1, x), 2 * x)
    states = torch.cat((y1, y2), dim=1)
    dy = state_derivative(states, x)
    assert torch.allclose(dy[:, 0:1], 2 * x)
    assert torch.allclose(dy[:, 1:2], 3 * x**2)


def test_reproducible_initialization():
    set_reproducibility(99)
    a = build_network(2, (0.0, 1.0), hidden_layers=(6,)).state_dict()
    set_reproducibility(99)
    b = build_network(2, (0.0, 1.0), hidden_layers=(6,)).state_dict()
    for key in a:
        assert torch.equal(a[key], b[key])


def _np_eq(x, y, p):
    return np.vstack((y[1], -np.ones_like(x)))


def _np_bc(ya, yb, p):
    return np.array([ya[0], yb[0]])


def _torch_eq(x, y, p):
    return torch.stack((y[1], -torch.ones_like(x)))


def _torch_bc(ya, yb, p):
    return torch.stack((ya[0], yb[0]))


def test_small_pinn_training_reproducible_and_accurate():
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
        hidden_layers=(16, 16),
        n_collocation=48,
        adam_epochs=600,
        adam_lr=2e-3,
        use_lbfgs=True,
        lbfgs_max_iter=80,
        bc_weight=10.0,
        seed=321,
        history_every=50,
        verbose=False,
        early_stopping_patience=None,
    )
    sol1 = solve(problem, method="pinn", pinn_config=config)
    sol2 = solve(problem, method="pinn", pinn_config=config)
    x = np.linspace(0, 1, 51)
    exact = 0.5 * x * (1 - x)
    err = np.max(np.abs(sol1.values("y", x) - exact))
    assert err < 2e-3
    assert np.allclose(sol1.values("y", x), sol2.values("y", x), atol=1e-12, rtol=1e-12)
    assert sol1.metadata["training_history"].final is not None

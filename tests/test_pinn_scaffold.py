"""Fast tests for the v0.2 PINN module scaffold."""

import pytest

from pinns4bvp.pinn import PINNConfig
from pinns4bvp.backends.pinn_backend import solve_with_pinn


def test_pinn_config_defaults_are_valid():
    config = PINNConfig()
    assert config.hidden_layers == (64, 64, 64)
    assert config.dtype == "float64"
    assert config.device == "cpu"


def test_pinn_backend_is_explicitly_unimplemented():
    with pytest.raises(NotImplementedError, match="PINN backend"):
        solve_with_pinn()

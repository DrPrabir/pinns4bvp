"""PINN unknown-parameter eigenvalue example for PINNs4BVP v0.5."""

from __future__ import annotations

import numpy as np
import torch

from pinns4bvp import BVPProblem, UnknownParameter, solve
from pinns4bvp.pinn import PINNConfig


def equations(x, y, p):
    k = p["k"]
    return np.vstack((y[1], -(k**2) * y[0]))


def boundary_conditions(ya, yb, p):
    return np.array((ya[0], yb[0], ya[1] - 1.0))


def pinn_equations(x, y, p):
    k = p["k"]
    return torch.stack((y[1], -(k**2) * y[0]))


def pinn_boundary_conditions(ya, yb, p):
    return torch.stack((ya[0], yb[0], ya[1] - 1.0))


def main():
    problem = BVPProblem(
        equations=equations,
        boundary_conditions=boundary_conditions,
        domain=(0.0, 1.0),
        n_equations=2,
        unknown_parameters={"k": UnknownParameter(3.0)},
        variable_names=("y", "yp"),
        name="PINN eigenvalue BVP",
        pinn_equations=pinn_equations,
        pinn_boundary_conditions=pinn_boundary_conditions,
    )

    config = PINNConfig(
        hidden_layers=(32, 32),
        n_collocation=96,
        adam_epochs=2500,
        adam_lr=1e-3,
        use_lbfgs=True,
        lbfgs_max_iter=200,
        bc_weight=20.0,
        seed=1234,
        verbose=True,
        early_stopping_patience=None,
    )

    sol = solve(problem, method="pinn", pinn_config=config)
    print("\n" + sol.summary())
    print(f"\nExact k       : {np.pi:.12f}")
    print(f"Recovered k   : {sol.parameters['k']:.12f}")
    print(f"Absolute error: {abs(sol.parameters['k'] - np.pi):.3e}")


if __name__ == "__main__":
    main()

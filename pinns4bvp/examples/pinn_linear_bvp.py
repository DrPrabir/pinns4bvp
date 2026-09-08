"""Linear BVP solved with both collocation and a PyTorch PINN.

    y'' = -1,  y(0)=0, y(1)=0

Exact solution: y = x(1-x)/2.
"""

import numpy as np
import torch

from pinns4bvp import BVPProblem, solve
from pinns4bvp.benchmark import compare_solutions
from pinns4bvp.pinn import PINNConfig


def equations(x, y, p):
    return np.vstack((y[1], -np.ones_like(x)))


def bc(ya, yb, p):
    return np.array([ya[0], yb[0]])


def pinn_equations(x, y, p):
    return torch.stack((y[1], -torch.ones_like(x)))


def pinn_bc(ya, yb, p):
    return torch.stack((ya[0], yb[0]))


def main():
    problem = BVPProblem(
        equations=equations,
        boundary_conditions=bc,
        pinn_equations=pinn_equations,
        pinn_boundary_conditions=pinn_bc,
        domain=(0.0, 1.0),
        n_equations=2,
        variable_names=("y", "yp"),
        name="Linear test BVP",
    )

    classical = solve(problem, method="collocation", tol=1e-10)
    config = PINNConfig(
        hidden_layers=(32, 32),
        n_collocation=80,
        adam_epochs=2000,
        adam_lr=1e-3,
        use_lbfgs=True,
        lbfgs_max_iter=150,
        seed=1234,
        verbose=True,
    )
    neural = solve(problem, method="pinn", pinn_config=config)

    metrics = compare_solutions(classical, neural, variable="y")
    print(neural.metadata["training_history"].summary())
    print(f"RMSE vs collocation: {metrics.rmse:.3e}")
    print(f"Max error          : {metrics.max_abs_error:.3e}")


if __name__ == "__main__":
    main()

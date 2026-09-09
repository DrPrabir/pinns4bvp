"""v0.4 benchmark example: numerical collocation vs PINN vs exact solution."""

import numpy as np
import torch

from pinns4bvp import BVPProblem, benchmark_problem
from pinns4bvp.pinn import PINNConfig


def equations(x, y, p):
    return np.vstack((y[1], -np.ones_like(x)))


def boundary_conditions(ya, yb, p):
    return np.array([ya[0], yb[0]])


def pinn_equations(x, y, p):
    return torch.stack((y[1], -torch.ones_like(x)))


def pinn_boundary_conditions(ya, yb, p):
    return torch.stack((ya[0], yb[0]))


def exact_solution(x):
    x = np.asarray(x, dtype=float)
    return np.vstack((0.5 * x * (1.0 - x), 0.5 - x))


def main():
    problem = BVPProblem(
        equations=equations,
        boundary_conditions=boundary_conditions,
        pinn_equations=pinn_equations,
        pinn_boundary_conditions=pinn_boundary_conditions,
        domain=(0.0, 1.0),
        n_equations=2,
        variable_names=("y", "yp"),
        name="Linear benchmark BVP",
    )

    config = PINNConfig(
        hidden_layers=(32, 32),
        n_collocation=80,
        adam_epochs=2000,
        adam_lr=1e-3,
        use_lbfgs=True,
        lbfgs_max_iter=150,
        bc_weight=10.0,
        seed=1234,
        deterministic=True,
        dtype="float64",
        device="cpu",
        verbose=False,
        early_stopping_patience=None,
    )

    report = benchmark_problem(
        problem,
        exact=exact_solution,
        variables=("y", "yp"),
        n_points=501,
        numerical_kwargs={"tol": 1e-10},
        pinn_config=config,
    )

    print(report.summary())

    metrics = report.metrics("pinn_vs_exact", "y")
    print(f"\nPINN y RMSE vs exact: {metrics.rmse:.3e}")

    report.plot("y")

    import matplotlib.pyplot as plt

    plt.show()


if __name__ == "__main__":
    main()

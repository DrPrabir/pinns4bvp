"""Experimental PINN warm-start continuation for PINNs4BVP v0.7.

The same linear solution family y=lambda*x is solved with a PINN. Each accepted
PINN model is used to initialize the next continuation point.
"""

import numpy as np
import torch

from pinns4bvp import BVPProblem, continue_parameter
from pinns4bvp.pinn import PINNConfig


def equations(x, y, p):
    return np.vstack((y[1], np.zeros_like(x)))


def boundary_conditions(ya, yb, p):
    return np.array((ya[0], yb[0] - p["lambda"]))


def pinn_equations(x, y, p):
    return torch.stack((y[1], torch.zeros_like(x)))


def pinn_boundary_conditions(ya, yb, p):
    return torch.stack((ya[0], yb[0] - p["lambda"]))


def main():
    problem = BVPProblem(
        equations,
        boundary_conditions,
        (0.0, 1.0),
        2,
        parameters={"lambda": 0.0},
        variable_names=("y", "yp"),
        pinn_equations=pinn_equations,
        pinn_boundary_conditions=pinn_boundary_conditions,
        name="PINN continuation example",
    )

    config = PINNConfig(
        hidden_layers=(16, 16),
        n_collocation=48,
        adam_epochs=500,
        adam_lr=2e-3,
        use_lbfgs=True,
        lbfgs_max_iter=80,
        bc_weight=10.0,
        seed=321,
        verbose=False,
        early_stopping_patience=None,
        device="cpu",
        dtype="float64",
    )

    family = continue_parameter(
        problem,
        "lambda",
        [0.0, 0.5, 1.0],
        method="pinn",
        pinn_config=config,
    )

    print(family.summary())
    print("\nWarm-start status")
    for point in family.requested_results:
        if point.solution is not None:
            print(
                f"lambda={point.value:g}  accepted={point.accepted}  "
                f"warm_start={point.solution.metadata['warm_start_used']}"
            )


if __name__ == "__main__":
    main()

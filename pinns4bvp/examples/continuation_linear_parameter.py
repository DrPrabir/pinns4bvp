"""Simple natural parameter continuation example for PINNs4BVP v0.7.

Solve y'' = 0 with y(0)=0 and y(1)=lambda while lambda is deliberately
varied through a sequence of fixed values. The exact family is y=lambda*x.
"""

import numpy as np

from pinns4bvp import BVPProblem, continue_parameter


def equations(x, y, p):
    return np.vstack((y[1], np.zeros_like(x)))


def boundary_conditions(ya, yb, p):
    return np.array((ya[0], yb[0] - p["lambda"]))


def main():
    problem = BVPProblem(
        equations,
        boundary_conditions,
        (0.0, 1.0),
        2,
        parameters={"lambda": 0.0},
        variable_names=("y", "yp"),
        name="Linear continuation example",
    )

    family = continue_parameter(
        problem,
        "lambda",
        np.linspace(0.0, 2.0, 5),
        solve_kwargs={"tol": 1e-9},
    )

    print(family.summary())
    parameter_values, midpoint_values = family.track("y", 0.5)
    print("\nMidpoint response")
    for lam, value in zip(parameter_values, midpoint_values):
        print(f"lambda={lam:5.2f}  y(0.5)={value:.8f}")


if __name__ == "__main__":
    main()

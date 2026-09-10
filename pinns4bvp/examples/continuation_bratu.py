"""Natural parameter continuation for the nonlinear Bratu BVP."""

import numpy as np

from pinns4bvp import BVPProblem, ContinuationConfig, continue_parameter


def equations(x, y, p):
    return np.vstack((y[1], -p["lambda"] * np.exp(y[0])))


def boundary_conditions(ya, yb, p):
    return np.array((ya[0], yb[0]))


def initial_guess(x):
    return np.vstack((0.1 * np.sin(np.pi * x), 0.1 * np.pi * np.cos(np.pi * x)))


def main():
    problem = BVPProblem(
        equations,
        boundary_conditions,
        (0.0, 1.0),
        2,
        parameters={"lambda": 0.25},
        variable_names=("y", "yp"),
        name="Bratu continuation",
    )

    family = continue_parameter(
        problem,
        "lambda",
        [0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        initial_guess=initial_guess,
        solve_kwargs={"tol": 1e-7, "max_nodes": 10000},
        config=ContinuationConfig(max_step=0.5),
    )

    print(family.summary())
    lam, midpoint = family.track("y", 0.5)
    print("\nBratu midpoint response")
    for p, ymid in zip(lam, midpoint):
        print(f"lambda={p:4.2f}  y(0.5)={ymid:.8f}")


if __name__ == "__main__":
    main()

"""Continuation with a simultaneously solved unknown eigen-parameter.

The fixed parameter mu is varied while k is solved from
    y'' + mu*k^2*y = 0,
    y(0)=0, y(1)=0, y'(0)=1.
The first positive branch satisfies k = pi/sqrt(mu).
"""

import numpy as np

from pinns4bvp import BVPProblem, UnknownParameter, continue_parameter


def equations(x, y, p):
    return np.vstack((y[1], -(p["mu"] * p["k"] ** 2) * y[0]))


def boundary_conditions(ya, yb, p):
    return np.array((ya[0], yb[0], ya[1] - 1.0))


def initial_guess(x):
    return np.vstack((np.sin(np.pi * x) / np.pi, np.cos(np.pi * x)))


def main():
    problem = BVPProblem(
        equations,
        boundary_conditions,
        (0.0, 1.0),
        2,
        parameters={"mu": 1.0},
        unknown_parameters={"k": UnknownParameter(3.0)},
        variable_names=("y", "yp"),
        name="Eigenvalue continuation",
    )

    family = continue_parameter(
        problem,
        "mu",
        [1.0, 1.25, 1.5, 2.0],
        initial_guess=initial_guess,
        solve_kwargs={"tol": 1e-9},
    )

    print(family.summary())
    print("\nSolved eigen-parameter")
    for point in family.requested_results:
        if point.accepted:
            k = point.solution.parameters["k"]
            exact = np.pi / np.sqrt(point.value)
            print(
                f"mu={point.value:4.2f}  k={k:.10f}  "
                f"exact={exact:.10f}  error={abs(k-exact):.3e}"
            )


if __name__ == "__main__":
    main()

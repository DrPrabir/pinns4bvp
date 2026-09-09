"""Unknown-parameter eigenvalue BVP example for PINNs4BVP v0.5.

Solve
    y'' + k^2 y = 0,   0 <= x <= 1
with
    y(0) = 0,
    y(1) = 0,
    y'(0) = 1.

The normalization y'(0)=1 removes the trivial solution. The first positive
solution has k = pi and y(x) = sin(pi*x)/pi.
"""

from __future__ import annotations

import numpy as np

from pinns4bvp import BVPProblem, UnknownParameter, solve


def equations(x, y, p):
    k = p["k"]
    return np.vstack((y[1], -(k**2) * y[0]))


def boundary_conditions(ya, yb, p):
    return np.array((ya[0], yb[0], ya[1] - 1.0))


def initial_guess(x):
    k0 = 3.0
    return np.vstack((np.sin(k0 * x) / k0, np.cos(k0 * x)))


def main():
    problem = BVPProblem(
        equations=equations,
        boundary_conditions=boundary_conditions,
        domain=(0.0, 1.0),
        n_equations=2,
        unknown_parameters={"k": UnknownParameter(3.0, "first eigen-wavenumber")},
        variable_names=("y", "yp"),
        name="Eigenvalue BVP",
    )

    sol = solve(problem, guess=initial_guess, tol=1e-9)
    print(sol.summary())
    print(f"\nExact k       : {np.pi:.12f}")
    print(f"Recovered k   : {sol.parameters['k']:.12f}")
    print(f"Absolute error: {abs(sol.parameters['k'] - np.pi):.3e}")


if __name__ == "__main__":
    main()

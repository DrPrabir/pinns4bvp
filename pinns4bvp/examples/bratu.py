"""Bratu nonlinear BVP.

Solve y'' + lambda*exp(y) = 0, y(0)=y(1)=0.
A nonzero initial guess can be used to explore different nonlinear branches.
"""

import numpy as np

from pinns4bvp import BVPProblem, solve


LAMBDA = 1.0


def equations(x, y, p):
    return np.vstack((y[1], -p["lambda"] * np.exp(y[0])))


def boundary_conditions(ya, yb, p):
    return np.array([ya[0], yb[0]])


def guess(x):
    y = np.zeros((2, x.size))
    y[0] = 0.5 * np.sin(np.pi * x)
    y[1] = 0.5 * np.pi * np.cos(np.pi * x)
    return y


def main():
    problem = BVPProblem(
        equations=equations,
        boundary_conditions=boundary_conditions,
        domain=(0.0, 1.0),
        n_equations=2,
        parameters={"lambda": LAMBDA},
        variable_names=("y", "y_prime"),
        name="Bratu problem",
    )

    solution = solve(problem, guess=guess, tol=1e-7)
    print(solution.summary())
    print(f"y(0.5)           : {solution.values('y', 0.5):.8f}")

    import matplotlib.pyplot as plt

    solution.plot("y")
    plt.show()


if __name__ == "__main__":
    main()

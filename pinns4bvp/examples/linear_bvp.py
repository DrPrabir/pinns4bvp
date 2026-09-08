"""Linear verification problem.

Solve y'' = -1 on [0, 1], y(0)=y(1)=0.
Exact solution: y = x(1-x)/2.
"""

import numpy as np

from pinns4bvp import BVPProblem, solve


def equations(x, y, p):
    return np.vstack((y[1], -np.ones_like(x)))


def boundary_conditions(ya, yb, p):
    return np.array([ya[0], yb[0]])


def main():
    problem = BVPProblem(
        equations=equations,
        boundary_conditions=boundary_conditions,
        domain=(0.0, 1.0),
        n_equations=2,
        variable_names=("y", "y_prime"),
        name="Linear BVP",
    )

    solution = solve(problem, tol=1e-8)
    print(solution.summary())

    x = np.linspace(0.0, 1.0, 201)
    y_exact = 0.5 * x * (1.0 - x)
    y_num = solution.values("y", x)
    print(f"Max error         : {np.max(np.abs(y_num - y_exact)):.3e}")

    import matplotlib.pyplot as plt

    ax = solution.plot("y")
    ax.plot(x, y_exact, "--", label="exact")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()

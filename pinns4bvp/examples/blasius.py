"""Blasius boundary-layer equation on a truncated semi-infinite domain.

f''' + 0.5 f f'' = 0
f(0)=0, f'(0)=0, f'(eta_max)=1

For v0.1 the user chooses eta_max manually. Automatic far-field extension is planned
for a later release.
"""

import numpy as np

from pinns4bvp import BVPProblem, solve


ETA_MAX = 10.0


def equations(eta, y, p):
    f, fp, fpp = y
    return np.vstack((fp, fpp, -0.5 * f * fpp))


def boundary_conditions(ya, yb, p):
    return np.array([
        ya[0],
        ya[1],
        yb[1] - 1.0,
    ])


def guess(eta):
    f = eta + np.exp(-eta) - 1.0
    fp = 1.0 - np.exp(-eta)
    fpp = np.exp(-eta)
    return np.vstack((f, fp, fpp))


def main():
    problem = BVPProblem(
        equations=equations,
        boundary_conditions=boundary_conditions,
        domain=(0.0, ETA_MAX),
        n_equations=3,
        variable_names=("f", "fp", "fpp"),
        name="Blasius boundary layer",
    )

    solution = solve(
        problem,
        n_mesh=80,
        mesh_kind="quadratic",
        guess=guess,
        tol=1e-7,
        max_nodes=20000,
    )

    print(solution.summary())
    print(f"f''(0)           : {solution.wall_value('fpp'):.8f}")

    import matplotlib.pyplot as plt

    solution.plot("fp")
    plt.show()


if __name__ == "__main__":
    main()

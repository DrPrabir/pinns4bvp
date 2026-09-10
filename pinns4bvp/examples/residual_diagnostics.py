"""v0.6 example: mesh, reusable guess, and residual diagnostics."""

import numpy as np

from pinns4bvp import BVPProblem, solve
from pinns4bvp.mesh import MeshConfig


def equations(x, y, p):
    return np.vstack((y[1], -np.ones_like(x)))


def bc(ya, yb, p):
    return np.array((ya[0], yb[0]))


problem = BVPProblem(
    equations,
    bc,
    (0.0, 1.0),
    2,
    variable_names=("y", "yp"),
    name="v0.6 residual diagnostics example",
)

mesh = MeshConfig(n_nodes=40, kind="chebyshev")
sol = solve(problem, mesh=mesh, tol=1e-9)

print(sol.summary())
print()
print(sol.metadata["initial_mesh_quality"].summary())
print()
print(sol.residual_report(n_points=201).summary())

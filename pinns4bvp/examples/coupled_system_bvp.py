"""v0.8: simultaneous nonlinear second-order system."""

import numpy as np

from pinns4bvp import DependentVariable, Equation, HigherOrderBVP, d2


u = DependentVariable("u", order=2)
v = DependentVariable("v", order=2)

# Exact reference: u=1/(1+x), v=1/(1+x)^2.
problem = HigherOrderBVP(
    equations=[
        Equation(d2(u) - 2.0 * u * v, 0.0),
        Equation(d2(v) - 6.0 * u**2 * v, 0.0),
    ],
    boundary_conditions=[
        u.at(0.0) == 1.0,
        u.at(1.0) == 0.5,
        v.at(0.0) == 1.0,
        v.at(1.0) == 0.25,
    ],
    domain=(0.0, 1.0),
    name="Coupled nonlinear BVP",
)


def guess(x):
    return np.vstack((
        1.0 / (1.0 + x),
        -1.0 / (1.0 + x) ** 2,
        1.0 / (1.0 + x) ** 2,
        -2.0 / (1.0 + x) ** 3,
    ))


solution = problem.solve(guess=guess, tol=1e-9)
print(solution.summary())
print("states:", solution.variable_names)

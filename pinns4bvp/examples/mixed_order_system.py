"""v0.8: mixed-order system with coupled highest derivatives."""

import numpy as np

from pinns4bvp import DependentVariable, Equation, HigherOrderBVP, d, d2, d3


u = DependentVariable("u", order=3)
v = DependentVariable("v", order=2)

# Exact: u=x^3, v=x^2. The compiler solves the 2x2 leading-derivative
# system for u''' and v'' automatically.
problem = HigherOrderBVP(
    equations=[
        Equation(d3(u) + d2(v), 8.0),
        Equation(d3(u) - d2(v), 4.0),
    ],
    boundary_conditions=[
        u.at(0.0) == 0.0,
        d(u).at(0.0) == 0.0,
        u.at(1.0) == 1.0,
        v.at(0.0) == 0.0,
        v.at(1.0) == 1.0,
    ],
    domain=(0.0, 1.0),
)

solution = problem.solve(
    guess=lambda x: np.vstack((x**3, 3*x**2, 6*x, x**2, 2*x)),
    tol=1e-10,
)
print(solution.summary())
print("states:", solution.variable_names)

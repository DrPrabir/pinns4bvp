"""v0.8: compact higher-order BVP with default y(x)."""

import numpy as np

from pinns4bvp import Equation, HigherOrderBVP, Parameter, d, d2, y


a = Parameter("a", value=2.0)
b = Parameter("b", value=3.0)
c = Parameter("c", value=1.0)

problem = HigherOrderBVP(
    equation=Equation(d2(y) + a * d(y) + b * y, c),
    boundary_conditions=[
        y.at(0.0) == 0.0,
        y.at(1.0) == 1.0,
    ],
    domain=(0.0, 1.0),
    name="Second-order linear BVP",
)

solution = problem.solve(tol=1e-9)
print(solution.summary())
print("state names:", solution.variable_names)
print("y(0.5):", float(solution.values("y", 0.5)))
print("y'(0.5):", float(solution.derivative("y", order=1, x=0.5)))

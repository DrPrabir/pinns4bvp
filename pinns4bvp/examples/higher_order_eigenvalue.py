"""v0.8: eigenvalue BVP written directly in second-order form."""

import numpy as np

from pinns4bvp import Equation, HigherOrderBVP, Parameter, d, d2, y


k = Parameter("k", initial=3.0, unknown=True)

problem = HigherOrderBVP(
    equation=Equation(d2(y) + k**2 * y, 0.0),
    boundary_conditions=[
        y.at(0.0) == 0.0,
        y.at(1.0) == 0.0,
        d(y).at(0.0) == 1.0,
    ],
    domain=(0.0, 1.0),
    name="Second-order eigenvalue BVP",
)

solution = problem.solve(
    guess=lambda x: np.vstack((np.sin(np.pi*x)/np.pi, np.cos(np.pi*x))),
    tol=1e-10,
)
print(solution.summary())
print(f"|k-pi| = {abs(solution.parameter('k') - np.pi):.3e}")

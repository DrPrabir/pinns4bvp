"""v0.8: nonlinear BVP with an exact reference solution."""

import numpy as np

from pinns4bvp import Equation, HigherOrderBVP, Parameter, d2, y


mu = Parameter("mu", value=2.0)

# y'' = 2 mu^2 y^3, y(0)=1, y(1)=1/(1+mu)
# Exact y = 1/(1+mu*x)
problem = HigherOrderBVP(
    equation=Equation(d2(y), 2.0 * mu**2 * y**3),
    boundary_conditions=[
        y.at(0.0) == 1.0,
        y.at(1.0) == 1.0 / (1.0 + mu),
    ],
    domain=(0.0, 1.0),
    name="Nonlinear cubic BVP",
)

solution = problem.solve(tol=1e-9)
x_eval = np.linspace(0.0, 1.0, 201)
exact = 1.0 / (1.0 + 2.0 * x_eval)
error = np.max(np.abs(solution.values("y", x_eval) - exact))
print(solution.summary())
print(f"maximum error: {error:.3e}")

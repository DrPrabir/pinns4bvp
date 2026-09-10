"""v0.8: one mathematical equation definition used by the PINN backend."""

import numpy as np

from pinns4bvp import Equation, HigherOrderBVP, d2, y
from pinns4bvp.pinn import PINNConfig


# y'' = -1, y(0)=y(1)=0; exact y=x(1-x)/2.
# No separate PyTorch equation or BC callbacks are required.
problem = HigherOrderBVP(
    equation=Equation(d2(y), -1.0),
    boundary_conditions=[y.at(0.0) == 0.0, y.at(1.0) == 0.0],
    domain=(0.0, 1.0),
)

config = PINNConfig(
    hidden_layers=(32, 32),
    n_collocation=80,
    adam_epochs=2000,
    adam_lr=1e-3,
    use_lbfgs=True,
    lbfgs_max_iter=150,
    bc_weight=10.0,
    seed=1234,
    device="auto",
    dtype="float32",
    verbose=True,
)

solution = problem.solve(method="pinn", pinn_config=config)
x_eval = np.linspace(0.0, 1.0, 201)
exact = 0.5 * x_eval * (1.0 - x_eval)
rmse = np.sqrt(np.mean((solution.values("y", x_eval) - exact) ** 2))
print(solution.summary())
print("device:", solution.metadata["device_resolved"])
print(f"RMSE: {rmse:.3e}")

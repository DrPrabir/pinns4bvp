"""Loss construction for first-order BVP PINNs."""

from __future__ import annotations

from dataclasses import dataclass

from pinns4bvp.pinn.autodiff import state_derivative


@dataclass(slots=True)
class PINNLoss:
    total: object
    ode: object
    bc: object


def compute_pinn_loss(
    problem,
    model,
    x_collocation,
    *,
    ode_weight: float,
    bc_weight: float,
    unknown_values=None,
):
    """Compute ODE residual, boundary residual, and weighted total loss."""

    import torch

    if problem.pinn_equations is None or problem.pinn_boundary_conditions is None:
        raise ValueError(
            "PINN solving requires pinn_equations and pinn_boundary_conditions "
            "that operate on PyTorch tensors."
        )

    y_pred = model(x_collocation)
    dy_dx = state_derivative(y_pred, x_collocation)

    x_flat = x_collocation[:, 0]
    y_state = y_pred.T
    rhs = problem.evaluate_pinn_equations(x_flat, y_state, unknown_values).T
    ode_residual = dy_dx - rhs
    ode_loss = torch.mean(ode_residual.square())

    xa = torch.as_tensor([[problem.a]], dtype=x_collocation.dtype, device=x_collocation.device)
    xb = torch.as_tensor([[problem.b]], dtype=x_collocation.dtype, device=x_collocation.device)
    ya = model(xa)[0]
    yb = model(xb)[0]
    bc_residual = problem.evaluate_pinn_boundary_conditions(ya, yb, unknown_values)
    bc_loss = torch.mean(bc_residual.square())

    total = ode_weight * ode_loss + bc_weight * bc_loss
    return PINNLoss(total=total, ode=ode_loss, bc=bc_loss)

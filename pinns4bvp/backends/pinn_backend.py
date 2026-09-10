"""PyTorch PINN backend for general two-point boundary-value problems."""

from __future__ import annotations

import numpy as np

from pinns4bvp.diagnostics.convergence import ConvergenceReport
from pinns4bvp.pinn.autodiff import derivative
from pinns4bvp.pinn.config import PINNConfig
from pinns4bvp.pinn.network import build_network
from pinns4bvp.pinn.reproducibility import set_reproducibility
from pinns4bvp.pinn.trainer import train_pinn
from pinns4bvp.solution import BVPSolution


def _make_evaluator(model, problem, config, *, resolved_device: str):
    import torch

    dtype = {"float32": torch.float32, "float64": torch.float64}[config.dtype]
    device = torch.device(resolved_device)

    def evaluate(x, *, derivative: int = 0):
        arr = np.asarray(x, dtype=float)
        scalar = arr.ndim == 0
        flat = arr.reshape(-1)
        xt = torch.as_tensor(flat[:, None], dtype=dtype, device=device)
        if derivative > 0:
            xt = xt.detach().clone().requires_grad_(True)
        y = model(xt)
        if derivative > 0:
            columns = [
                derivative_fn(y[:, i : i + 1], xt, order=derivative)
                for i in range(y.shape[1])
            ]
            y = torch.cat(columns, dim=1)
        values = y.detach().cpu().numpy().T
        if scalar:
            return values[:, 0]
        return values.reshape(problem.n_equations, *arr.shape)

    derivative_fn = derivative
    return evaluate


def solve_with_pinn(
    problem,
    *,
    config: PINNConfig | None = None,
    n_output: int = 201,
    warm_start=None,
):
    """Solve ``problem`` with a PINN, including scalar unknown parameters.

    A previous compatible PINN ``BVPSolution`` may be supplied as
    ``warm_start`` to reuse network weights and solved unknown-parameter values.
    """

    import torch

    if problem.pinn_equations is None or problem.pinn_boundary_conditions is None:
        raise ValueError(
            "PINN backend requires problem.pinn_equations and "
            "problem.pinn_boundary_conditions."
        )
    if n_output < 2:
        raise ValueError("n_output must be at least 2")

    config = PINNConfig() if config is None else config
    set_reproducibility(config.seed, deterministic=config.deterministic)
    model = build_network(
        problem.n_equations,
        problem.domain,
        hidden_layers=config.hidden_layers,
        activation=config.activation,
    )
    warm_state = None
    unknown_initial = None
    warm_start_used = False
    if warm_start is not None:
        previous_model = getattr(warm_start, "metadata", {}).get("model")
        if previous_model is None:
            raise ValueError("PINN warm_start solution does not contain a trained model")
        if warm_start.problem.n_equations != problem.n_equations:
            raise ValueError("PINN warm_start has incompatible n_equations")
        if tuple(warm_start.problem.domain) != tuple(problem.domain):
            raise ValueError("PINN warm_start has an incompatible domain")
        warm_state = {
            key: value.detach().cpu().clone()
            for key, value in previous_model.state_dict().items()
        }
        unknown_initial = {
            name: warm_start.parameters[name]
            for name in problem.unknown_parameter_names
            if name in warm_start.parameters
        }
        warm_start_used = True

    result = train_pinn(
        problem,
        model,
        config,
        warm_start_state=warm_state,
        unknown_initial=unknown_initial,
    )
    model = result.model
    model.eval()
    evaluator = _make_evaluator(
        model,
        problem,
        config,
        resolved_device=result.device,
    )

    x = np.linspace(problem.a, problem.b, n_output)
    y = evaluator(x)

    final = result.history.final
    final_ode = float("inf") if final is None else final.ode_loss
    final_bc = float("inf") if final is None else final.bc_loss
    success = bool(
        np.isfinite(final_ode)
        and np.isfinite(final_bc)
        and max(final_ode, final_bc) <= max(1e-6, config.loss_tolerance * 100.0)
    )
    message = (
        "PINN training converged to configured diagnostic threshold"
        if success
        else "PINN training completed; inspect training diagnostics and residuals"
    )
    report = ConvergenceReport(
        success=success,
        status=0 if success else 1,
        message=message,
        n_iterations=result.history.adam_steps + result.history.lbfgs_evaluations,
        n_nodes=config.n_collocation,
        max_rms_residual=float(np.sqrt(final_ode)) if np.isfinite(final_ode) else None,
        bc_residual_inf=float(np.sqrt(final_bc)) if np.isfinite(final_bc) else float("inf"),
    )

    resolved = problem.parameter_mapping(result.unknown_parameters.floats())
    parameter_values = {name: float(value) for name, value in resolved.items()}

    return BVPSolution(
        problem=problem,
        x=x,
        y=y,
        diagnostics=report,
        parameters=parameter_values,
        _evaluator=evaluator,
        metadata={
            "backend": "pinn",
            "training_history": result.history,
            "config": config,
            "model": model,
            "torch_version": torch.__version__,
            "device_requested": config.device,
            "device_resolved": result.device,
            "dtype": config.dtype,
            "warm_start_used": warm_start_used,
        },
    )

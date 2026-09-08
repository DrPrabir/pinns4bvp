"""Robust Adam + L-BFGS training for BVP PINNs."""

from __future__ import annotations

from dataclasses import dataclass

from pinns4bvp.diagnostics.pinn_training import PINNTrainingHistory
from pinns4bvp.pinn.losses import compute_pinn_loss
from pinns4bvp.pinn.reproducibility import set_reproducibility


@dataclass(slots=True)
class PINNTrainingResult:
    model: object
    history: PINNTrainingHistory
    x_collocation: object


def _torch_dtype(name: str):
    import torch

    return {"float32": torch.float32, "float64": torch.float64}[name]


def _make_collocation(problem, config):
    import torch

    dtype = _torch_dtype(config.dtype)
    device = torch.device(config.device)
    if config.sampling == "uniform":
        x = torch.linspace(problem.a, problem.b, config.n_collocation, dtype=dtype, device=device)
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed)
        r = torch.rand(config.n_collocation, generator=generator, dtype=dtype)
        x = problem.a + (problem.b - problem.a) * r
        x, _ = torch.sort(x)
        x = x.to(device)
    return x[:, None].detach().clone().requires_grad_(True)


def _gradient_norm(model) -> float:
    import torch

    sq = torch.zeros((), dtype=torch.float64)
    for parameter in model.parameters():
        if parameter.grad is not None:
            sq += parameter.grad.detach().double().norm(2).square().cpu()
    return float(torch.sqrt(sq))


def train_pinn(problem, model, config) -> PINNTrainingResult:
    """Train ``model`` for ``problem`` with Adam followed by optional L-BFGS."""

    import torch

    set_reproducibility(config.seed, deterministic=config.deterministic)
    dtype = _torch_dtype(config.dtype)
    device = torch.device(config.device)
    model = model.to(device=device, dtype=dtype)
    x_collocation = _make_collocation(problem, config)
    history = PINNTrainingHistory()

    best = float("inf")
    stale = 0

    if config.adam_epochs > 0:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.adam_lr)
        for epoch in range(1, config.adam_epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            losses = compute_pinn_loss(
                problem,
                model,
                x_collocation,
                ode_weight=config.ode_weight,
                bc_weight=config.bc_weight,
            )
            losses.total.backward()
            grad_norm = _gradient_norm(model)
            if config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            history.adam_steps = epoch

            total_value = float(losses.total.detach().cpu())
            if epoch == 1 or epoch % config.history_every == 0 or epoch == config.adam_epochs:
                history.append(
                    step=epoch,
                    stage="adam",
                    total_loss=total_value,
                    ode_loss=float(losses.ode.detach().cpu()),
                    bc_loss=float(losses.bc.detach().cpu()),
                    grad_norm=grad_norm,
                )

            if config.verbose and (epoch == 1 or epoch % config.print_every == 0):
                print(
                    f"Adam {epoch:6d} | total={total_value:.3e} "
                    f"ode={float(losses.ode.detach().cpu()):.3e} "
                    f"bc={float(losses.bc.detach().cpu()):.3e}"
                )

            if total_value <= config.loss_tolerance:
                history.stop_reason = "loss_tolerance reached during Adam"
                break

            if total_value < best - config.early_stopping_min_delta:
                best = total_value
                stale = 0
            else:
                stale += 1

            if (
                config.early_stopping_patience is not None
                and stale >= config.early_stopping_patience
            ):
                history.stop_reason = "Adam early stopping (no improvement)"
                break

    if config.use_lbfgs and config.lbfgs_max_iter > 0:
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=config.lbfgs_lr,
            max_iter=config.lbfgs_max_iter,
            history_size=config.lbfgs_history_size,
            line_search_fn=config.lbfgs_line_search,
            tolerance_grad=1e-12,
            tolerance_change=1e-14,
        )
        closure_calls = 0

        def closure():
            nonlocal closure_calls
            optimizer.zero_grad(set_to_none=True)
            losses = compute_pinn_loss(
                problem,
                model,
                x_collocation,
                ode_weight=config.ode_weight,
                bc_weight=config.bc_weight,
            )
            losses.total.backward()
            closure_calls += 1
            history.lbfgs_evaluations = closure_calls
            if closure_calls == 1 or closure_calls % config.history_every == 0:
                history.append(
                    step=closure_calls,
                    stage="lbfgs",
                    total_loss=float(losses.total.detach().cpu()),
                    ode_loss=float(losses.ode.detach().cpu()),
                    bc_loss=float(losses.bc.detach().cpu()),
                    grad_norm=_gradient_norm(model),
                )
            return losses.total

        optimizer.step(closure)

        final_losses = compute_pinn_loss(
            problem,
            model,
            x_collocation,
            ode_weight=config.ode_weight,
            bc_weight=config.bc_weight,
        )
        history.append(
            step=closure_calls,
            stage="lbfgs-final",
            total_loss=float(final_losses.total.detach().cpu()),
            ode_loss=float(final_losses.ode.detach().cpu()),
            bc_loss=float(final_losses.bc.detach().cpu()),
            grad_norm=None,
        )
        if float(final_losses.total.detach().cpu()) <= config.loss_tolerance:
            history.stop_reason = "loss_tolerance reached after L-BFGS"
        elif history.stop_reason is None:
            history.stop_reason = "Adam + L-BFGS completed"
    elif history.stop_reason is None:
        history.stop_reason = "Adam completed"

    return PINNTrainingResult(model=model, history=history, x_collocation=x_collocation)

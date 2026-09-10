"""Robust Adam + L-BFGS training for BVP PINNs."""

from __future__ import annotations

from dataclasses import dataclass

from pinns4bvp.diagnostics.pinn_training import PINNTrainingHistory
from pinns4bvp.pinn.device import resolve_device
from pinns4bvp.pinn.losses import compute_pinn_loss
from pinns4bvp.pinn.parameters import PINNUnknownParameterSet
from pinns4bvp.pinn.reproducibility import set_reproducibility


@dataclass(slots=True)
class PINNTrainingResult:
    model: object
    history: PINNTrainingHistory
    x_collocation: object
    unknown_parameters: PINNUnknownParameterSet
    device: str


def _torch_dtype(name: str):
    import torch

    return {"float32": torch.float32, "float64": torch.float64}[name]


def _make_collocation(problem, config, *, device: str):
    import torch

    dtype = _torch_dtype(config.dtype)
    target = torch.device(device)
    if config.sampling == "uniform":
        x = torch.linspace(
            problem.a,
            problem.b,
            config.n_collocation,
            dtype=dtype,
            device=target,
        )
    else:
        # Generate deterministically on CPU then transfer to the requested device.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed)
        r = torch.rand(config.n_collocation, generator=generator, dtype=dtype)
        x = problem.a + (problem.b - problem.a) * r
        x, _ = torch.sort(x)
        x = x.to(target)
    return x[:, None].detach().clone().requires_grad_(True)


def _gradient_norm(parameters) -> float:
    import torch

    sq = torch.zeros((), dtype=torch.float64)
    for parameter in parameters:
        if parameter.grad is not None:
            sq += parameter.grad.detach().double().norm(2).square().cpu()
    return float(torch.sqrt(sq))


def train_pinn(problem, model, config, *, warm_start_state=None, unknown_initial=None) -> PINNTrainingResult:
    """Train ``model`` and any unknown scalar parameters simultaneously.

    ``warm_start_state`` may contain a compatible PyTorch ``state_dict`` from
    a previous PINN solution. ``unknown_initial`` may override initial values
    of unknown BVP parameters, which is useful during continuation.
    """

    import torch

    set_reproducibility(config.seed, deterministic=config.deterministic)
    dtype = _torch_dtype(config.dtype)
    resolved_device = resolve_device(config.device, dtype=config.dtype)
    device = torch.device(resolved_device)
    model = model.to(device=device, dtype=dtype)
    if warm_start_state is not None:
        try:
            model.load_state_dict(warm_start_state, strict=True)
        except RuntimeError as exc:
            raise ValueError(
                "PINN warm start is incompatible with the current network architecture"
            ) from exc
    unknown = PINNUnknownParameterSet(
        problem, dtype=dtype, device=device, initial_values=unknown_initial
    )
    x_collocation = _make_collocation(problem, config, device=resolved_device)
    history = PINNTrainingHistory()

    trainables = list(model.parameters()) + unknown.trainable()
    best = float("inf")
    stale = 0

    def parameter_values():
        return unknown.mapping() if problem.n_unknown_parameters else None

    def parameter_floats():
        return unknown.floats()

    if config.adam_epochs > 0:
        optimizer = torch.optim.Adam(trainables, lr=config.adam_lr)
        for epoch in range(1, config.adam_epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            losses = compute_pinn_loss(
                problem,
                model,
                x_collocation,
                ode_weight=config.ode_weight,
                bc_weight=config.bc_weight,
                unknown_values=parameter_values(),
            )
            losses.total.backward()
            grad_norm = _gradient_norm(trainables)
            if config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(trainables, config.grad_clip_norm)
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
                    parameters=parameter_floats(),
                )

            if config.verbose and (epoch == 1 or epoch % config.print_every == 0):
                suffix = ""
                if problem.n_unknown_parameters:
                    suffix = " | " + ", ".join(
                        f"{name}={value:.8g}" for name, value in parameter_floats().items()
                    )
                print(
                    f"Adam {epoch:6d} | total={total_value:.3e} "
                    f"ode={float(losses.ode.detach().cpu()):.3e} "
                    f"bc={float(losses.bc.detach().cpu()):.3e}{suffix}"
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
            trainables,
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
                unknown_values=parameter_values(),
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
                    grad_norm=_gradient_norm(trainables),
                    parameters=parameter_floats(),
                )
            return losses.total

        optimizer.step(closure)

        final_losses = compute_pinn_loss(
            problem,
            model,
            x_collocation,
            ode_weight=config.ode_weight,
            bc_weight=config.bc_weight,
            unknown_values=parameter_values(),
        )
        history.append(
            step=closure_calls,
            stage="lbfgs-final",
            total_loss=float(final_losses.total.detach().cpu()),
            ode_loss=float(final_losses.ode.detach().cpu()),
            bc_loss=float(final_losses.bc.detach().cpu()),
            grad_norm=None,
            parameters=parameter_floats(),
        )
        if float(final_losses.total.detach().cpu()) <= config.loss_tolerance:
            history.stop_reason = "loss_tolerance reached after L-BFGS"
        elif history.stop_reason is None:
            history.stop_reason = "Adam + L-BFGS completed"
    elif history.stop_reason is None:
        if history.final is not None and problem.n_unknown_parameters:
            final = compute_pinn_loss(
                problem,
                model,
                x_collocation,
                ode_weight=config.ode_weight,
                bc_weight=config.bc_weight,
                unknown_values=parameter_values(),
            )
            history.append(
                step=history.adam_steps,
                stage="adam-final",
                total_loss=float(final.total.detach().cpu()),
                ode_loss=float(final.ode.detach().cpu()),
                bc_loss=float(final.bc.detach().cpu()),
                grad_norm=None,
                parameters=parameter_floats(),
            )
        history.stop_reason = "Adam completed"

    return PINNTrainingResult(
        model=model,
        history=history,
        x_collocation=x_collocation,
        unknown_parameters=unknown,
        device=resolved_device,
    )

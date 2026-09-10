"""Natural continuation in one fixed scalar BVP parameter."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Callable, Iterable

import numpy as np

from pinns4bvp.continuation.config import ContinuationConfig
from pinns4bvp.continuation.family import ContinuationPoint, SolutionFamily
from pinns4bvp.parameters import UnknownParameter
from pinns4bvp.solver import solve


def _normalize_values(values: Iterable[float]) -> tuple[float, ...]:
    seq = tuple(float(v) for v in values)
    if not seq:
        raise ValueError("values must contain at least one parameter value")
    if not all(np.isfinite(v) for v in seq):
        raise ValueError("continuation values must be finite")
    if len(seq) > 1:
        diff = np.diff(seq)
        if not (np.all(diff > 0) or np.all(diff < 0)):
            raise ValueError("continuation values must be strictly monotonic")
    return seq


def _problem_at_value(
    base_problem,
    parameter: str,
    value: float,
    *,
    previous_solution=None,
    reuse_unknown_parameters: bool,
):
    fixed = dict(base_problem.parameters)
    fixed[parameter] = float(value)

    unknown = dict(base_problem.unknown_parameters)
    if previous_solution is not None and reuse_unknown_parameters:
        warmed = {}
        for name, spec in unknown.items():
            initial = previous_solution.parameters.get(name, spec.initial)
            warmed[name] = UnknownParameter(initial, spec.description)
        unknown = warmed

    return replace(base_problem, parameters=fixed, unknown_parameters=unknown)


def continue_parameter(
    problem,
    parameter: str,
    values: Iterable[float],
    *,
    method: str = "collocation",
    config: ContinuationConfig | None = None,
    solve_kwargs: dict | None = None,
    pinn_config=None,
    initial_guess=None,
    accept_solution: Callable[[object], bool] | None = None,
) -> SolutionFamily:
    """Continue a BVP through a monotonic sequence of fixed parameter values.

    The previous accepted state is automatically reused as the next classical
    initial guess. For PINN continuation, compatible network weights are warm
    started when ``ContinuationConfig.pinn_warm_start`` is enabled.

    ``accept_solution`` can override the default ``solution.success`` criterion,
    which is useful for experimental PINN workflows with custom residual
    tolerances.
    """

    if not isinstance(parameter, str) or not parameter:
        raise ValueError("parameter must be a non-empty string")
    if parameter in problem.unknown_parameters:
        raise ValueError(
            f"'{parameter}' is declared unknown; continuation requires a fixed parameter"
        )
    if parameter not in problem.parameters:
        raise KeyError(
            f"continuation parameter '{parameter}' is not present in problem.parameters"
        )

    targets = _normalize_values(values)
    config = ContinuationConfig() if config is None else config
    method = method.lower()
    if method not in {"collocation", "scipy", "pinn", "torch"}:
        raise ValueError("method must be 'collocation'/'scipy' or 'pinn'/'torch'")

    family = SolutionFamily(
        problem=problem,
        parameter_name=parameter,
        requested_values=targets,
        method="pinn" if method in {"pinn", "torch"} else "collocation",
        metadata={"config": config},
    )
    base_kwargs = dict(solve_kwargs or {})
    if initial_guess is not None and "guess" in base_kwargs:
        raise ValueError("initial_guess and solve_kwargs['guess'] cannot both be supplied")

    accept = (lambda sol: bool(sol.success)) if accept_solution is None else accept_solution

    def attempt(value, *, requested, source_solution, source_value, retry_level):
        local_problem = _problem_at_value(
            problem,
            parameter,
            value,
            previous_solution=source_solution,
            reuse_unknown_parameters=config.reuse_unknown_parameters,
        )
        kwargs = dict(base_kwargs)
        warm_used = False
        if method in {"pinn", "torch"}:
            kwargs["pinn_config"] = pinn_config
            if source_solution is not None and config.pinn_warm_start:
                kwargs["pinn_warm_start"] = source_solution
                warm_used = True
            # A classical guess has no meaning for the PINN backend.
            kwargs.pop("guess", None)
        else:
            if source_solution is not None:
                kwargs["guess"] = source_solution
            elif initial_guess is not None:
                kwargs["guess"] = initial_guess

        start = perf_counter()
        try:
            sol = solve(local_problem, method=method, **kwargs)
            elapsed = perf_counter() - start
            accepted = bool(accept(sol))
            message = sol.message
            residual = None
            if config.record_residuals:
                try:
                    residual = sol.residual_report(n_points=config.residual_points)
                except Exception:
                    residual = None
        except Exception as exc:  # continuation should record solver failures, not crash
            elapsed = perf_counter() - start
            sol = None
            accepted = False
            message = f"{type(exc).__name__}: {exc}"
            residual = None

        point = ContinuationPoint(
            value=float(value),
            solution=sol,
            accepted=accepted,
            requested=bool(requested),
            elapsed_seconds=float(elapsed),
            retry_level=int(retry_level),
            source_value=None if source_value is None else float(source_value),
            warm_start_used=warm_used,
            message=message,
            residual_report=residual,
        )
        family.points.append(point)
        return point

    def advance(source_solution, source_value, target, *, requested, retry_level):
        # Proactively limit very large jumps.
        if (
            source_solution is not None
            and config.max_step is not None
            and abs(target - source_value) > config.max_step
        ):
            direction = 1.0 if target > source_value else -1.0
            intermediate = source_value + direction * config.max_step
            inter_sol, inter_val, reached = advance(
                source_solution,
                source_value,
                intermediate,
                requested=False,
                retry_level=retry_level,
            )
            if not reached:
                return inter_sol, inter_val, False
            return advance(
                inter_sol,
                inter_val,
                target,
                requested=requested,
                retry_level=retry_level,
            )

        point = attempt(
            target,
            requested=requested,
            source_solution=source_solution,
            source_value=source_value,
            retry_level=retry_level,
        )
        if point.accepted:
            return point.solution, float(target), True

        can_retry = (
            config.adaptive
            and source_solution is not None
            and retry_level < config.max_retries
            and abs(target - source_value) > config.min_step
        )
        if not can_retry:
            return source_solution, source_value, False

        midpoint = source_value + config.reduction_factor * (target - source_value)
        if abs(midpoint - source_value) < config.min_step:
            return source_solution, source_value, False

        new_sol, new_val, reached_mid = advance(
            source_solution,
            source_value,
            midpoint,
            requested=False,
            retry_level=retry_level + 1,
        )
        progressed = new_sol is not source_solution or (
            new_val is not None and source_value is not None and abs(new_val - source_value) > 0
        )
        if not reached_mid and not progressed:
            return source_solution, source_value, False

        # Retry the original target from the closest accepted recovery point.
        return advance(
            new_sol,
            new_val,
            target,
            requested=requested,
            retry_level=retry_level + 1,
        )

    previous_solution = None
    previous_value = None

    for target in targets:
        previous_solution, previous_value, reached = advance(
            previous_solution,
            previous_value,
            target,
            requested=True,
            retry_level=0,
        )
        if not reached and config.stop_on_failure:
            break

    family.metadata["last_accepted_value"] = previous_value
    return family

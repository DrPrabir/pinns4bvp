"""Solution-family containers for parameter continuation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from pinns4bvp.continuation.diagnostics import ContinuationDiagnostics


@dataclass(slots=True)
class ContinuationPoint:
    """One attempted continuation point."""

    value: float
    solution: object | None
    accepted: bool
    requested: bool
    elapsed_seconds: float
    retry_level: int = 0
    source_value: float | None = None
    warm_start_used: bool = False
    message: str = ""
    residual_report: object | None = None
    retained: bool = True

    @property
    def success(self) -> bool:
        return self.accepted


@dataclass(slots=True)
class SolutionFamily:
    """A parameter-indexed continuation result with configurable retention."""

    problem: object
    parameter_name: str
    requested_values: tuple[float, ...]
    method: str
    points: list[ContinuationPoint] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def save_policy(self) -> str:
        return str(self.metadata.get("save", "all"))

    @property
    def requested_results(self) -> tuple[ContinuationPoint, ...]:
        """Final recorded result for each requested parameter value."""

        latest: dict[float, ContinuationPoint] = {}
        for point in self.points:
            if point.requested:
                latest[float(point.value)] = point
        return tuple(latest[value] for value in self.requested_values if value in latest)

    @property
    def successful_points(self) -> tuple[ContinuationPoint, ...]:
        return tuple(point for point in self.points if point.accepted)

    @property
    def failed_points(self) -> tuple[ContinuationPoint, ...]:
        return tuple(point for point in self.requested_results if not point.accepted)

    @property
    def retained_points(self) -> tuple[ContinuationPoint, ...]:
        return tuple(
            point
            for point in self.points
            if point.retained and point.solution is not None
        )

    @property
    def solutions(self) -> tuple[object, ...]:
        """Retained accepted requested solutions.

        With ``save='final'`` this contains only the final requested solution;
        use ``requested_results`` for lightweight status at every target.
        """

        return tuple(
            point.solution
            for point in self.requested_results
            if point.accepted and point.solution is not None and point.retained
        )

    @property
    def values(self) -> np.ndarray:
        return np.asarray(
            [
                point.value
                for point in self.requested_results
                if point.accepted and point.solution is not None and point.retained
            ],
            dtype=float,
        )

    @property
    def final_point(self) -> ContinuationPoint | None:
        if not self.requested_values:
            return None
        target = float(self.requested_values[-1])
        matches = [p for p in self.requested_results if float(p.value) == target]
        return matches[-1] if matches else None

    @property
    def final_solution(self):
        point = self.final_point
        if point is None or not point.accepted or point.solution is None:
            return None
        return point.solution

    @property
    def success(self) -> bool:
        results = self.requested_results
        return len(results) == len(self.requested_values) and all(p.accepted for p in results)

    @property
    def diagnostics(self) -> ContinuationDiagnostics:
        requested_results = self.requested_results
        requested_converged = sum(point.accepted for point in requested_results)
        requested_failed = len(self.requested_values) - requested_converged
        intermediate = [point for point in self.points if not point.requested]
        return ContinuationDiagnostics(
            parameter_name=self.parameter_name,
            n_requested=len(self.requested_values),
            n_attempts=len(self.points),
            n_requested_converged=requested_converged,
            n_requested_failed=requested_failed,
            n_intermediate_attempts=len(intermediate),
            n_intermediate_converged=sum(point.accepted for point in intermediate),
            total_solve_time=float(sum(point.elapsed_seconds for point in self.points)),
        )

    @property
    def history(self) -> tuple[dict, ...]:
        """Lightweight continuation history retained under every save policy."""

        rows = []
        for point in self.points:
            residual = point.residual_report
            rows.append(
                {
                    "parameter": float(point.value),
                    "requested": bool(point.requested),
                    "accepted": bool(point.accepted),
                    "retry_level": int(point.retry_level),
                    "source_value": point.source_value,
                    "elapsed_seconds": float(point.elapsed_seconds),
                    "warm_start_used": bool(point.warm_start_used),
                    "retained": bool(point.retained and point.solution is not None),
                    "ode_rms": None if residual is None else residual.ode_rms,
                    "ode_max_abs": None if residual is None else residual.ode_max_abs,
                    "bc_max_abs": None if residual is None else residual.bc_max_abs,
                    "message": point.message,
                }
            )
        return tuple(rows)

    def summary(self) -> str:
        lines = [
            self.diagnostics.summary(),
            f"Save policy      : {self.save_policy}",
            "",
            "Requested values",
            "----------------",
        ]
        result_map = {float(point.value): point for point in self.requested_results}
        for value in self.requested_values:
            point = result_map.get(value)
            if point is None:
                status = "not attempted"
                message = ""
            elif point.accepted:
                status = "success"
                message = point.message
            else:
                status = "failed"
                message = point.message
            suffix = f" - {message}" if message else ""
            retained = " [retained]" if point is not None and point.solution is not None else ""
            lines.append(f"{value: .10g} : {status}{retained}{suffix}")
        return "\n".join(lines)

    def solution_at(self, value: float, *, atol: float = 1e-12, requested_only: bool = False):
        candidates = self.requested_results if requested_only else tuple(self.points)
        matches = [
            point
            for point in candidates
            if point.accepted
            and point.solution is not None
            and abs(point.value - value) <= atol
        ]
        if not matches:
            raise KeyError(
                f"no retained accepted solution found at {self.parameter_name}={value}; "
                f"save policy is '{self.save_policy}'"
            )
        return matches[-1].solution

    def evaluate(self, func: Callable, *, requested_only: bool = True):
        source = self.requested_results if requested_only else tuple(self.points)
        values = []
        outputs = []
        for point in source:
            if point.accepted and point.solution is not None:
                values.append(point.value)
                outputs.append(func(point.solution))
        return np.asarray(values, dtype=float), np.asarray(outputs)

    def track(self, variable: str | int, x: float, *, derivative: int = 0, requested_only: bool = True):
        return self.evaluate(
            lambda sol: float(np.asarray(sol.values(variable, x, derivative=derivative))),
            requested_only=requested_only,
        )

    def plot_profiles(self, variable: str | int = 0, *, derivative: int = 0, requested_only: bool = True, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        source = self.requested_results if requested_only else tuple(self.points)
        for point in source:
            if point.accepted and point.solution is not None:
                sol = point.solution
                idx = sol.problem.variable_index(variable)
                ax.plot(
                    sol.x,
                    sol.values(idx, derivative=derivative),
                    label=f"{self.parameter_name}={point.value:g}",
                )
        ax.set_xlabel("x")
        ax.set_ylabel(str(variable))
        ax.grid(True, alpha=0.25)
        ax.legend()
        return ax

    def plot_track(self, variable: str | int, x: float, *, derivative: int = 0, requested_only: bool = True, ax=None):
        import matplotlib.pyplot as plt

        p, response = self.track(
            variable,
            x,
            derivative=derivative,
            requested_only=requested_only,
        )
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(p, response, marker="o")
        ax.set_xlabel(self.parameter_name)
        ax.set_ylabel(f"{variable} at x={x:g}")
        ax.grid(True, alpha=0.25)
        return ax

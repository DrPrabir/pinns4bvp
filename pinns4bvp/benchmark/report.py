"""Structured benchmark report objects."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pinns4bvp.benchmark.metrics import ComparisonMetrics


@dataclass(slots=True)
class MethodRun:
    """One timed solution method within a benchmark."""

    name: str
    solution: object
    runtime_seconds: float | None

    @property
    def success(self) -> bool:
        return bool(self.solution.success)


@dataclass(slots=True)
class BenchmarkReport:
    """Numerical/PINN/exact benchmark results on one common grid."""

    problem: object
    x: np.ndarray
    variables: tuple[str, ...]
    numerical: MethodRun | None = None
    pinn: MethodRun | None = None
    exact: object | None = None
    comparisons: dict[str, dict[str, ComparisonMetrics]] = field(default_factory=dict)

    def metrics(self, comparison: str, variable: str | int = 0) -> ComparisonMetrics:
        idx = self.problem.variable_index(variable)
        name = self.problem.variable_names[idx]
        try:
            return self.comparisons[comparison][name]
        except KeyError as exc:
            available = tuple(self.comparisons)
            raise KeyError(
                f"comparison '{comparison}' for variable '{name}' is unavailable. "
                f"Available comparisons: {available}"
            ) from exc

    def summary(self) -> str:
        lines = [
            f"Benchmark: {self.problem.name}",
            "=" * max(20, len(self.problem.name) + 11),
            f"Grid points : {self.x.size}",
            f"Variables   : {', '.join(self.variables)}",
        ]
        if self.numerical is not None:
            lines.append(
                f"Numerical   : {'success' if self.numerical.success else 'failed'} "
                + (f"({self.numerical.runtime_seconds:.6f} s)" if self.numerical.runtime_seconds is not None else "(runtime not measured)")
            )
        if self.pinn is not None:
            lines.append(
                f"PINN        : {'success' if self.pinn.success else 'failed'} "
                + (f"({self.pinn.runtime_seconds:.6f} s)" if self.pinn.runtime_seconds is not None else "(runtime not measured)")
            )
        lines.append(f"Exact       : {'available' if self.exact is not None else 'not supplied'}")

        for comparison, by_variable in self.comparisons.items():
            lines.extend(["", comparison.replace("_", " ").title()])
            lines.append("variable      RMSE          MAE           Max abs       Relative L2")
            lines.append("-" * 69)
            for variable in self.variables:
                metrics = by_variable.get(variable)
                if metrics is None:
                    continue
                lines.append(
                    f"{variable:<12} "
                    f"{metrics.rmse:>11.3e} "
                    f"{metrics.mae:>13.3e} "
                    f"{metrics.max_abs_error:>13.3e} "
                    f"{metrics.relative_l2:>13.3e}"
                )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        data = {
            "problem": self.problem.name,
            "grid_points": int(self.x.size),
            "variables": list(self.variables),
            "runtime_seconds": {
                "numerical": None if self.numerical is None else self.numerical.runtime_seconds,
                "pinn": None if self.pinn is None else self.pinn.runtime_seconds,
            },
            "success": {
                "numerical": None if self.numerical is None else self.numerical.success,
                "pinn": None if self.pinn is None else self.pinn.success,
            },
            "comparisons": {},
        }
        for comparison, by_variable in self.comparisons.items():
            data["comparisons"][comparison] = {
                variable: metrics.as_dict()
                for variable, metrics in by_variable.items()
            }
        return data

    def plot(self, variable: str | int = 0, *, ax=None):
        from pinns4bvp.benchmark.plotting import plot_benchmark

        return plot_benchmark(self, variable=variable, ax=ax)

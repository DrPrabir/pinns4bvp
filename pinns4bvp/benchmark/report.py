"""Structured benchmark report objects."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pinns4bvp.benchmark.metrics import ComparisonMetrics, ParameterMetrics


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
    exact_parameters: dict[str, float] = field(default_factory=dict)
    comparisons: dict[str, dict[str, ComparisonMetrics]] = field(default_factory=dict)
    parameter_comparisons: dict[str, dict[str, ParameterMetrics]] = field(default_factory=dict)

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

    def parameter_metrics(self, comparison: str, name: str) -> ParameterMetrics:
        try:
            return self.parameter_comparisons[comparison][name]
        except KeyError as exc:
            raise KeyError(
                f"parameter comparison '{comparison}' for '{name}' is unavailable"
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

        for comparison, by_parameter in self.parameter_comparisons.items():
            lines.extend(["", "Parameter " + comparison.replace("_", " ").title()])
            lines.append("parameter     reference      estimate       Abs error      Relative error")
            lines.append("-" * 73)
            for name, metrics in by_parameter.items():
                lines.append(
                    f"{name:<12} "
                    f"{metrics.reference:>12.6g} "
                    f"{metrics.estimate:>13.6g} "
                    f"{metrics.abs_error:>14.3e} "
                    f"{metrics.relative_error:>14.3e}"
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
            "exact_parameters": dict(self.exact_parameters),
            "comparisons": {},
            "parameter_comparisons": {},
        }
        for comparison, by_variable in self.comparisons.items():
            data["comparisons"][comparison] = {
                variable: metrics.as_dict()
                for variable, metrics in by_variable.items()
            }
        for comparison, by_parameter in self.parameter_comparisons.items():
            data["parameter_comparisons"][comparison] = {
                name: metrics.as_dict()
                for name, metrics in by_parameter.items()
            }
        return data

    def plot(self, variable: str | int = 0, *, ax=None):
        from pinns4bvp.benchmark.plotting import plot_benchmark

        return plot_benchmark(self, variable=variable, ax=ax)

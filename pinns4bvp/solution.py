from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from pinns4bvp.diagnostics.convergence import ConvergenceReport
from pinns4bvp.problem import BVPProblem


@dataclass(slots=True)
class BVPSolution:
    problem: BVPProblem
    x: np.ndarray
    y: np.ndarray
    diagnostics: ConvergenceReport
    parameters: dict[str, float] = field(default_factory=dict)
    _raw_solution: object | None = None
    _evaluator: Callable | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.diagnostics.success

    @property
    def message(self) -> str:
        return self.diagnostics.message

    @property
    def variable_names(self) -> tuple[str, ...]:
        return tuple(self.problem.variable_names)

    @property
    def unknown_parameters(self) -> dict[str, float]:
        return {
            name: self.parameters[name]
            for name in self.problem.unknown_parameter_names
            if name in self.parameters
        }

    def parameter(self, name: str) -> float:
        try:
            return float(self.parameters[name])
        except KeyError as exc:
            raise KeyError(
                f"unknown parameter '{name}'. Available: {tuple(self.parameters)}"
            ) from exc

    def __call__(self, x, *, derivative: int = 0) -> np.ndarray:
        if derivative < 0:
            raise ValueError("derivative order must be >= 0")
        if self._evaluator is not None:
            return np.asarray(self._evaluator(x, derivative=derivative))
        if self._raw_solution is not None and hasattr(self._raw_solution, "sol"):
            return np.asarray(self._raw_solution.sol(x, nu=derivative))
        raise RuntimeError("this solution does not provide an evaluator")

    def values(self, variable: str | int, x=None, *, derivative: int = 0):
        idx = self.problem.variable_index(variable)
        if x is None:
            if derivative == 0:
                return self.y[idx]
            x = self.x
        values = self(x, derivative=derivative)
        return values[idx]

    def wall_value(
        self,
        variable: str | int,
        *,
        derivative: int = 0,
        side: str = "left",
    ) -> float:
        if side == "left":
            x = self.problem.a
        elif side == "right":
            x = self.problem.b
        else:
            raise ValueError("side must be 'left' or 'right'")
        return float(np.asarray(self.values(variable, x, derivative=derivative)))

    def summary(self) -> str:
        text = self.diagnostics.summary()
        if self.problem.unknown_parameter_names:
            lines = [text, "", "Solved parameters", "-----------------"]
            for name in self.problem.unknown_parameter_names:
                if name in self.parameters:
                    lines.append(f"{name:<16}: {self.parameters[name]:.12g}")
            return "\n".join(lines)
        return text


    def residual_report(self, *, x=None, n_points: int = 201):
        """Return independent ODE and boundary residual diagnostics."""

        from pinns4bvp.diagnostics.residuals import compute_residual_report

        return compute_residual_report(self, x=x, n_points=n_points)

    def residuals(self, *, x=None, n_points: int = 201):
        """Alias for :meth:`residual_report`."""

        return self.residual_report(x=x, n_points=n_points)

    def plot_residuals(self, *, x=None, n_points: int = 201, ax=None):
        """Plot absolute ODE residuals for all state equations."""

        import matplotlib.pyplot as plt

        report = self.residual_report(x=x, n_points=n_points)
        if ax is None:
            _, ax = plt.subplots()
        for i, name in enumerate(report.variable_names):
            ax.semilogy(report.x, np.maximum(np.abs(report.ode_residuals[i]), 1e-300), label=name)
        ax.set_xlabel("x")
        ax.set_ylabel("|ODE residual|")
        ax.grid(True, alpha=0.25)
        ax.legend()
        return ax

    def plot(self, variable: str | int = 0, *, derivative: int = 0, ax=None, **kwargs):
        import matplotlib.pyplot as plt

        idx = self.problem.variable_index(variable)
        if ax is None:
            _, ax = plt.subplots()
        y = self.values(idx, derivative=derivative)
        name = self.variable_names[idx]
        label = name if derivative == 0 else f"d^{derivative}({name})/dx^{derivative}"
        ax.plot(self.x, y, label=label, **kwargs)
        ax.set_xlabel("x")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
        ax.legend()
        return ax

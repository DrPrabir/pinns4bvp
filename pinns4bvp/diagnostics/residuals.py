"""Independent residual diagnostics for solved BVPs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ResidualReport:
    """ODE and boundary residual diagnostics evaluated on a common grid."""

    x: np.ndarray
    ode_residuals: np.ndarray
    bc_residuals: np.ndarray
    variable_names: tuple[str, ...]

    @property
    def ode_max_abs_by_equation(self) -> np.ndarray:
        return np.max(np.abs(self.ode_residuals), axis=1)

    @property
    def ode_mean_abs_by_equation(self) -> np.ndarray:
        return np.mean(np.abs(self.ode_residuals), axis=1)

    @property
    def ode_rms_by_equation(self) -> np.ndarray:
        return np.sqrt(np.mean(self.ode_residuals**2, axis=1))

    @property
    def ode_max_abs(self) -> float:
        return float(np.max(np.abs(self.ode_residuals)))

    @property
    def ode_rms(self) -> float:
        return float(np.sqrt(np.mean(self.ode_residuals**2)))

    @property
    def bc_max_abs(self) -> float:
        return float(np.max(np.abs(self.bc_residuals))) if self.bc_residuals.size else 0.0

    def summary(self) -> str:
        lines = [
            "Residual diagnostics",
            "--------------------",
            f"Grid points       : {self.x.size}",
            f"Global ODE RMS    : {self.ode_rms:.3e}",
            f"Global ODE max abs: {self.ode_max_abs:.3e}",
            f"BC max abs        : {self.bc_max_abs:.3e}",
            "",
            "Per-equation residuals",
            "----------------------",
        ]
        for i, name in enumerate(self.variable_names):
            lines.append(
                f"{name:<16} RMS={self.ode_rms_by_equation[i]:.3e}  "
                f"max={self.ode_max_abs_by_equation[i]:.3e}"
            )
        return "\n".join(lines)


def compute_residual_report(solution, *, x=None, n_points: int = 201) -> ResidualReport:
    """Evaluate ``y' - f(x, y, p)`` and boundary residuals independently.

    The same function works for classical and PINN solutions because it uses the
    backend-independent ``BVPSolution`` evaluator.
    """

    if x is None:
        if not isinstance(n_points, int) or n_points < 2:
            raise ValueError("n_points must be an integer >= 2")
        x = np.linspace(solution.problem.a, solution.problem.b, n_points)
    else:
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or x.size < 2:
            raise ValueError("x must be a 1D array with at least 2 points")
        if not np.all(np.isfinite(x)) or not np.all(np.diff(x) > 0):
            raise ValueError("x must be finite and strictly increasing")
        if x[0] < solution.problem.a or x[-1] > solution.problem.b:
            raise ValueError("diagnostic x grid must lie inside the problem domain")

    y = np.asarray(solution(x), dtype=float)
    dy = np.asarray(solution(x, derivative=1), dtype=float)
    unknown_values = {
        name: solution.parameters[name]
        for name in solution.problem.unknown_parameter_names
        if name in solution.parameters
    }
    rhs = solution.problem.evaluate_equations(
        x,
        y,
        unknown_values if unknown_values else None,
    )
    ode = dy - rhs

    ya = np.asarray(solution(solution.problem.a), dtype=float)
    yb = np.asarray(solution(solution.problem.b), dtype=float)
    bc = solution.problem.evaluate_boundary_conditions(
        ya,
        yb,
        unknown_values if unknown_values else None,
    )

    return ResidualReport(
        x=np.asarray(x, dtype=float),
        ode_residuals=np.asarray(ode, dtype=float),
        bc_residuals=np.asarray(bc, dtype=float),
        variable_names=tuple(solution.problem.variable_names),
    )

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pinns4bvp.diagnostics.convergence import ConvergenceReport
from pinns4bvp.problem import BVPProblem


@dataclass(slots=True)
class BVPSolution:
    problem: BVPProblem
    x: np.ndarray
    y: np.ndarray
    diagnostics: ConvergenceReport
    _raw_solution: object

    @property
    def success(self) -> bool:
        return self.diagnostics.success

    @property
    def message(self) -> str:
        return self.diagnostics.message

    @property
    def variable_names(self) -> tuple[str, ...]:
        return tuple(self.problem.variable_names)

    def __call__(self, x, *, derivative: int = 0) -> np.ndarray:
        if derivative < 0:
            raise ValueError("derivative order must be >= 0")
        return np.asarray(self._raw_solution.sol(x, nu=derivative))

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
        return self.diagnostics.summary()

    def plot(self, variable: str | int = 0, *, derivative: int = 0, ax=None, **kwargs):
        """Plot one solution component. Matplotlib is imported lazily."""
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

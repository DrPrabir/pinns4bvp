from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

Array = np.ndarray
ODEFunction = Callable[[Array, Array, Mapping[str, float]], Array]
BCFunction = Callable[[Array, Array, Mapping[str, float]], Array]


@dataclass(slots=True)
class BVPProblem:
    """Mathematical definition of a two-point boundary-value problem.

    The classical callbacks use NumPy arrays.  During the alpha PINN releases,
    optional ``pinn_equations`` and ``pinn_boundary_conditions`` callbacks may
    be supplied for PyTorch tensors.  This explicit separation preserves
    autograd without breaking the established NumPy/SciPy API.
    """

    equations: ODEFunction
    boundary_conditions: BCFunction
    domain: tuple[float, float]
    n_equations: int
    parameters: Mapping[str, float] = field(default_factory=dict)
    variable_names: Sequence[str] | None = None
    name: str = "Boundary-value problem"
    pinn_equations: Callable | None = None
    pinn_boundary_conditions: Callable | None = None

    def __post_init__(self) -> None:
        if not callable(self.equations):
            raise TypeError("equations must be callable")
        if not callable(self.boundary_conditions):
            raise TypeError("boundary_conditions must be callable")
        if self.pinn_equations is not None and not callable(self.pinn_equations):
            raise TypeError("pinn_equations must be callable or None")
        if self.pinn_boundary_conditions is not None and not callable(
            self.pinn_boundary_conditions
        ):
            raise TypeError("pinn_boundary_conditions must be callable or None")

        if len(self.domain) != 2:
            raise ValueError("domain must be a tuple (a, b)")
        a, b = map(float, self.domain)
        if not np.isfinite(a) or not np.isfinite(b):
            raise ValueError("domain endpoints must be finite")
        if not a < b:
            raise ValueError("domain must satisfy a < b")
        self.domain = (a, b)

        if not isinstance(self.n_equations, int) or self.n_equations < 1:
            raise ValueError("n_equations must be a positive integer")

        self.parameters = dict(self.parameters)
        for key, value in self.parameters.items():
            if not isinstance(key, str) or not key:
                raise ValueError("parameter names must be non-empty strings")
            if not np.isscalar(value):
                raise ValueError(f"parameter '{key}' must be scalar")

        if self.variable_names is None:
            self.variable_names = tuple(f"y{i}" for i in range(self.n_equations))
        else:
            names = tuple(self.variable_names)
            if len(names) != self.n_equations:
                raise ValueError(
                    "variable_names length must equal n_equations "
                    f"({self.n_equations})"
                )
            if len(set(names)) != len(names):
                raise ValueError("variable_names must be unique")
            if any((not isinstance(name, str)) or (not name) for name in names):
                raise ValueError("variable_names must contain non-empty strings")
            self.variable_names = names

    @property
    def a(self) -> float:
        return self.domain[0]

    @property
    def b(self) -> float:
        return self.domain[1]

    def variable_index(self, variable: str | int) -> int:
        if isinstance(variable, int):
            if 0 <= variable < self.n_equations:
                return variable
            raise IndexError(f"variable index {variable} is out of range")
        try:
            return tuple(self.variable_names).index(variable)
        except ValueError as exc:
            raise KeyError(
                f"unknown variable '{variable}'. Available: {tuple(self.variable_names)}"
            ) from exc

    def evaluate_equations(self, x: Array, y: Array) -> Array:
        values = np.asarray(self.equations(x, y, self.parameters), dtype=float)
        if values.shape != y.shape:
            raise ValueError(
                "equations must return an array with the same shape as y; "
                f"expected {y.shape}, got {values.shape}"
            )
        return values

    def evaluate_boundary_conditions(self, ya: Array, yb: Array) -> Array:
        values = np.asarray(
            self.boundary_conditions(ya, yb, self.parameters), dtype=float
        ).reshape(-1)
        if values.size != self.n_equations:
            raise ValueError(
                "boundary_conditions must return exactly n_equations residuals "
                f"({self.n_equations}); got {values.size}"
            )
        return values

    def evaluate_pinn_equations(self, x, y):
        if self.pinn_equations is None:
            raise ValueError("pinn_equations was not supplied for this problem")
        values = self.pinn_equations(x, y, self.parameters)
        if tuple(values.shape) != tuple(y.shape):
            raise ValueError(
                "pinn_equations must return a tensor with the same shape as y; "
                f"expected {tuple(y.shape)}, got {tuple(values.shape)}"
            )
        return values

    def evaluate_pinn_boundary_conditions(self, ya, yb):
        if self.pinn_boundary_conditions is None:
            raise ValueError("pinn_boundary_conditions was not supplied for this problem")
        values = self.pinn_boundary_conditions(ya, yb, self.parameters).reshape(-1)
        if values.numel() != self.n_equations:
            raise ValueError(
                "pinn_boundary_conditions must return exactly n_equations residuals "
                f"({self.n_equations}); got {values.numel()}"
            )
        return values

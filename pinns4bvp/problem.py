from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

from pinns4bvp.parameters import UnknownParameter

Array = np.ndarray
ParameterMapping = Mapping[str, object]
ODEFunction = Callable[[Array, Array, ParameterMapping], Array]
BCFunction = Callable[[Array, Array, ParameterMapping], Array]


@dataclass(slots=True)
class BVPProblem:
    """Mathematical definition of a two-point boundary-value problem.

    The state is written as a first-order system ``y' = f(x, y, p)``. Fixed
    scalar parameters are supplied with ``parameters``. Scalar quantities that
    must be solved simultaneously with the state are supplied with
    ``unknown_parameters`` as :class:`~pinns4bvp.UnknownParameter` objects.

    For a system with ``n`` state equations and ``k`` unknown parameters, the
    boundary callback must return exactly ``n + k`` residuals. This matches the
    dimensionality required by collocation BVP solvers for unknown parameters.

    Classical callbacks operate on NumPy arrays. Optional ``pinn_equations``
    and ``pinn_boundary_conditions`` callbacks operate on PyTorch tensors so
    autograd is preserved by the PINN backend.
    """

    equations: ODEFunction
    boundary_conditions: BCFunction
    domain: tuple[float, float]
    n_equations: int
    parameters: Mapping[str, float] = field(default_factory=dict)
    unknown_parameters: Mapping[str, UnknownParameter] = field(default_factory=dict)
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

        fixed = dict(self.parameters)
        for key, value in fixed.items():
            if not isinstance(key, str) or not key:
                raise ValueError("parameter names must be non-empty strings")
            if not np.isscalar(value):
                raise ValueError(f"parameter '{key}' must be scalar")
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"parameter '{key}' must be finite")
            fixed[key] = value
        self.parameters = fixed

        unknown = dict(self.unknown_parameters)
        for key, spec in unknown.items():
            if not isinstance(key, str) or not key:
                raise ValueError("unknown parameter names must be non-empty strings")
            if key in fixed:
                raise ValueError(
                    f"parameter '{key}' cannot be both fixed and unknown"
                )
            if not isinstance(spec, UnknownParameter):
                raise TypeError(
                    f"unknown_parameters['{key}'] must be an UnknownParameter"
                )
        self.unknown_parameters = unknown

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

    @property
    def n_unknown_parameters(self) -> int:
        return len(self.unknown_parameters)

    @property
    def unknown_parameter_names(self) -> tuple[str, ...]:
        return tuple(self.unknown_parameters)

    @property
    def n_boundary_residuals(self) -> int:
        return self.n_equations + self.n_unknown_parameters

    @property
    def initial_unknown_values(self) -> np.ndarray:
        return np.asarray(
            [self.unknown_parameters[name].initial for name in self.unknown_parameter_names],
            dtype=float,
        )

    def parameter_mapping(self, unknown_values=None) -> dict[str, object]:
        """Return fixed and unknown parameters as one callback mapping.

        ``unknown_values`` may be a sequence in declared unknown-parameter
        order or a mapping keyed by unknown-parameter name. Values are kept as
        objects rather than forcibly cast to float so PyTorch scalar tensors
        can flow through PINN callbacks without breaking autograd.
        """

        values: dict[str, object] = dict(self.parameters)
        names = self.unknown_parameter_names
        if not names:
            return values

        if unknown_values is None:
            resolved = {
                name: self.unknown_parameters[name].initial for name in names
            }
        elif isinstance(unknown_values, Mapping):
            missing = [name for name in names if name not in unknown_values]
            extra = [name for name in unknown_values if name not in self.unknown_parameters]
            if missing:
                raise ValueError(f"missing unknown parameter values: {missing}")
            if extra:
                raise ValueError(f"unexpected unknown parameter values: {extra}")
            resolved = {name: unknown_values[name] for name in names}
        else:
            seq = list(unknown_values)
            if len(seq) != len(names):
                raise ValueError(
                    f"expected {len(names)} unknown parameter values, got {len(seq)}"
                )
            resolved = dict(zip(names, seq))

        values.update(resolved)
        return values


    def with_parameters(self, **updates):
        """Return a copy with updated fixed scalar parameters.

        This convenience method is useful for parameter studies and does not
        mutate the original problem.
        """

        from dataclasses import replace

        fixed = dict(self.parameters)
        fixed.update(updates)
        return replace(self, parameters=fixed)

    def continue_parameter(self, parameter: str, values, **kwargs):
        """Run one-parameter natural continuation from this problem.

        This is a convenience wrapper around :func:`pinns4bvp.continue_parameter`.
        """

        from pinns4bvp.continuation import continue_parameter

        return continue_parameter(self, parameter, values, **kwargs)

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

    def evaluate_equations(self, x: Array, y: Array, unknown_values=None) -> Array:
        p = self.parameter_mapping(unknown_values)
        values = np.asarray(self.equations(x, y, p), dtype=float)
        if values.shape != y.shape:
            raise ValueError(
                "equations must return an array with the same shape as y; "
                f"expected {y.shape}, got {values.shape}"
            )
        return values

    def evaluate_boundary_conditions(
        self, ya: Array, yb: Array, unknown_values=None
    ) -> Array:
        p = self.parameter_mapping(unknown_values)
        values = np.asarray(self.boundary_conditions(ya, yb, p), dtype=float).reshape(-1)
        if values.size != self.n_boundary_residuals:
            raise ValueError(
                "boundary_conditions must return exactly n_equations + "
                "n_unknown_parameters residuals "
                f"({self.n_boundary_residuals}); got {values.size}"
            )
        return values

    def evaluate_pinn_equations(self, x, y, unknown_values=None):
        if self.pinn_equations is None:
            raise ValueError("pinn_equations was not supplied for this problem")
        p = self.parameter_mapping(unknown_values)
        values = self.pinn_equations(x, y, p)
        if tuple(values.shape) != tuple(y.shape):
            raise ValueError(
                "pinn_equations must return a tensor with the same shape as y; "
                f"expected {tuple(y.shape)}, got {tuple(values.shape)}"
            )
        return values

    def evaluate_pinn_boundary_conditions(self, ya, yb, unknown_values=None):
        if self.pinn_boundary_conditions is None:
            raise ValueError("pinn_boundary_conditions was not supplied for this problem")
        p = self.parameter_mapping(unknown_values)
        values = self.pinn_boundary_conditions(ya, yb, p).reshape(-1)
        if values.numel() != self.n_boundary_residuals:
            raise ValueError(
                "pinn_boundary_conditions must return exactly n_equations + "
                "n_unknown_parameters residuals "
                f"({self.n_boundary_residuals}); got {values.numel()}"
            )
        return values

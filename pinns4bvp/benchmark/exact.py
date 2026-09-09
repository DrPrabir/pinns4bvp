"""Exact/reference solution adapters used by the benchmark runner."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np


class ExactReference:
    """Normalize exact-solution specifications to a common interface.

    ``exact`` may be either:

    * a callable ``exact(x)`` returning the complete state with shape
      ``(n_equations, n_points)`` (or ``(n_equations,)`` for scalar ``x``), or
    * a mapping from variable name/index to a callable returning that variable.
    """

    def __init__(self, problem, exact):
        if exact is None:
            raise ValueError("exact reference cannot be None")
        if not callable(exact) and not isinstance(exact, Mapping):
            raise TypeError("exact must be callable or a mapping of variables to callables")
        self.problem = problem
        self._exact = exact

    def has_variable(self, variable: str | int) -> bool:
        if callable(self._exact):
            return True
        idx = self.problem.variable_index(variable)
        name = self.problem.variable_names[idx]
        return name in self._exact or idx in self._exact

    def values(self, variable: str | int, x):
        idx = self.problem.variable_index(variable)
        name = self.problem.variable_names[idx]
        arr_x = np.asarray(x, dtype=float)

        if callable(self._exact):
            values = np.asarray(self._exact(arr_x), dtype=float)
            if arr_x.ndim == 0:
                if values.shape == (self.problem.n_equations,):
                    return values[idx]
                if values.shape == (self.problem.n_equations, 1):
                    return values[idx, 0]
            expected = (self.problem.n_equations, *arr_x.shape)
            if values.shape != expected:
                raise ValueError(
                    "exact callable must return the complete state with shape "
                    f"{expected}; got {values.shape}"
                )
            return values[idx]

        key = name if name in self._exact else idx
        if key not in self._exact:
            raise KeyError(f"exact reference was not supplied for variable '{name}'")
        fn = self._exact[key]
        if not callable(fn):
            raise TypeError(f"exact reference for variable '{name}' must be callable")
        values = np.asarray(fn(arr_x), dtype=float)
        if values.shape != arr_x.shape:
            raise ValueError(
                f"exact reference for variable '{name}' must return shape "
                f"{arr_x.shape}; got {values.shape}"
            )
        return values

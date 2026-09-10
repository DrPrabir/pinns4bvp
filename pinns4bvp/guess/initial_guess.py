from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from pinns4bvp.problem import BVPProblem


@dataclass(frozen=True, slots=True)
class InterpolatedGuess:
    """Reusable state guess defined on a source mesh and interpolated as needed."""

    x: np.ndarray
    y: np.ndarray

    def __post_init__(self) -> None:
        x = np.asarray(self.x, dtype=float).copy()
        y = np.asarray(self.y, dtype=float).copy()
        if x.ndim != 1 or x.size < 2:
            raise ValueError("source x must be a 1D array with at least 2 points")
        if not np.all(np.isfinite(x)) or not np.all(np.diff(x) > 0):
            raise ValueError("source x must be finite and strictly increasing")
        if y.ndim != 2 or y.shape[1] != x.size:
            raise ValueError("source y must have shape (n_equations, len(x))")
        if not np.all(np.isfinite(y)):
            raise ValueError("source y must contain only finite values")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)

    def __call__(self, x_new: np.ndarray) -> np.ndarray:
        x_new = np.asarray(x_new, dtype=float)
        return np.vstack([np.interp(x_new, self.x, row) for row in self.y])


def guess_from_solution(solution) -> InterpolatedGuess:
    """Create an interpolating initial guess from a previous ``BVPSolution``."""

    if not hasattr(solution, "x") or not hasattr(solution, "y"):
        raise TypeError("solution must provide x and y arrays")
    return InterpolatedGuess(solution.x, solution.y)


def _guess_from_mapping(problem: BVPProblem, x: np.ndarray, mapping: Mapping) -> np.ndarray:
    rows = []
    missing = []
    for index, name in enumerate(problem.variable_names):
        if name in mapping:
            spec = mapping[name]
        elif index in mapping:
            spec = mapping[index]
        else:
            missing.append(name)
            continue
        if callable(spec):
            row = np.asarray(spec(x), dtype=float)
        else:
            value = float(spec)
            row = np.full_like(x, value, dtype=float)
        if row.shape != x.shape:
            raise ValueError(
                f"guess for variable '{name}' must return shape {x.shape}; got {row.shape}"
            )
        rows.append(row)
    if missing:
        raise ValueError(f"mapping guess is missing variables: {missing}")
    return np.vstack(rows)


def create_initial_guess(
    problem: BVPProblem,
    x: np.ndarray,
    guess=None,
) -> np.ndarray:
    """Create and validate an initial solution guess on mesh ``x``.

    Accepted forms are:

    - ``None`` or ``"zeros"``;
    - a NumPy-compatible array with shape ``(n_equations, n_nodes)``;
    - a callable ``guess(x)``;
    - a :class:`InterpolatedGuess`;
    - a previous solution object exposing ``x`` and ``y``;
    - ``(x_source, y_source)`` for interpolation;
    - a mapping from variable names (or indices) to scalars/callables.
    """

    if guess is None or (isinstance(guess, str) and guess.lower() == "zeros"):
        y = np.zeros((problem.n_equations, x.size), dtype=float)
    elif isinstance(guess, str):
        raise ValueError("unknown guess string; supported string is 'zeros'")
    elif isinstance(guess, InterpolatedGuess):
        y = guess(x)
    elif isinstance(guess, Mapping):
        y = _guess_from_mapping(problem, x, guess)
    elif hasattr(guess, "x") and hasattr(guess, "y"):
        y = guess_from_solution(guess)(x)
    elif isinstance(guess, tuple) and len(guess) == 2:
        y = InterpolatedGuess(guess[0], guess[1])(x)
    elif callable(guess):
        y = np.asarray(guess(x), dtype=float)
    else:
        y = np.asarray(guess, dtype=float)

    expected = (problem.n_equations, x.size)
    if y.shape != expected:
        raise ValueError(f"initial guess must have shape {expected}; got {y.shape}")
    if not np.all(np.isfinite(y)):
        raise ValueError("initial guess must contain only finite values")
    return np.asarray(y, dtype=float)

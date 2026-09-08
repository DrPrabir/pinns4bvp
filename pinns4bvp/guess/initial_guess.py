from __future__ import annotations

from collections.abc import Callable

import numpy as np

from pinns4bvp.problem import BVPProblem


def create_initial_guess(
    problem: BVPProblem,
    x: np.ndarray,
    guess: str | np.ndarray | Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Create and validate the initial solution guess on mesh ``x``."""
    if guess is None or (isinstance(guess, str) and guess == "zeros"):
        y = np.zeros((problem.n_equations, x.size), dtype=float)
    elif isinstance(guess, str):
        raise ValueError("unknown guess string; v0.1 supports only 'zeros'")
    elif callable(guess):
        y = np.asarray(guess(x), dtype=float)
    else:
        y = np.asarray(guess, dtype=float)

    expected = (problem.n_equations, x.size)
    if y.shape != expected:
        raise ValueError(
            f"initial guess must have shape {expected}; got {y.shape}"
        )
    if not np.all(np.isfinite(y)):
        raise ValueError("initial guess must contain only finite values")
    return y

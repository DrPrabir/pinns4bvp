from __future__ import annotations

import numpy as np
from scipy.integrate import solve_bvp as scipy_solve_bvp

from pinns4bvp.problem import BVPProblem


def solve_with_scipy(
    problem: BVPProblem,
    mesh: np.ndarray,
    initial_guess: np.ndarray,
    *,
    tol: float = 1e-5,
    bc_tol: float | None = None,
    max_nodes: int = 10000,
    verbose: int = 0,
):
    """Solve ``problem`` using SciPy's collocation BVP backend.

    If ``problem`` declares unknown parameters, SciPy solves their values
    simultaneously with the state using its native ``p`` interface.
    """
    if tol <= 0:
        raise ValueError("tol must be positive")
    if bc_tol is not None and bc_tol <= 0:
        raise ValueError("bc_tol must be positive when provided")
    if max_nodes < mesh.size:
        raise ValueError("max_nodes cannot be smaller than the initial mesh size")
    if verbose not in (0, 1, 2):
        raise ValueError("verbose must be 0, 1, or 2")

    if problem.n_unknown_parameters:
        def fun(x, y, p):
            return problem.evaluate_equations(x, y, p)

        def bc(ya, yb, p):
            return problem.evaluate_boundary_conditions(ya, yb, p)

        return scipy_solve_bvp(
            fun,
            bc,
            mesh,
            initial_guess,
            p=problem.initial_unknown_values,
            tol=tol,
            bc_tol=bc_tol,
            max_nodes=max_nodes,
            verbose=verbose,
        )

    def fun(x, y):
        return problem.evaluate_equations(x, y)

    def bc(ya, yb):
        return problem.evaluate_boundary_conditions(ya, yb)

    return scipy_solve_bvp(
        fun,
        bc,
        mesh,
        initial_guess,
        tol=tol,
        bc_tol=bc_tol,
        max_nodes=max_nodes,
        verbose=verbose,
    )

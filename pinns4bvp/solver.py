from __future__ import annotations

import numpy as np

from pinns4bvp.backends.scipy_backend import solve_with_scipy
from pinns4bvp.diagnostics.convergence import build_convergence_report
from pinns4bvp.guess.initial_guess import create_initial_guess
from pinns4bvp.mesh.initial_mesh import create_initial_mesh, validate_mesh
from pinns4bvp.problem import BVPProblem
from pinns4bvp.solution import BVPSolution


def solve(
    problem: BVPProblem,
    *,
    method: str = "collocation",
    pinn_config=None,
    mesh: np.ndarray | None = None,
    n_mesh: int = 50,
    mesh_kind: str = "uniform",
    guess=None,
    tol: float = 1e-5,
    bc_tol: float | None = None,
    max_nodes: int = 10000,
    verbose: int = 0,
    raise_on_failure: bool = False,
) -> BVPSolution:
    """Solve a :class:`BVPProblem` with a selected backend.

    ``method='collocation'`` preserves the v0.1 SciPy behavior.
    ``method='pinn'`` activates the optional PyTorch backend.
    """

    if not isinstance(problem, BVPProblem):
        raise TypeError("problem must be an instance of BVPProblem")

    method = method.lower()
    if method in {"pinn", "torch"}:
        from pinns4bvp.backends.pinn_backend import solve_with_pinn

        solution = solve_with_pinn(problem, config=pinn_config)
        if raise_on_failure and not solution.success:
            raise RuntimeError(solution.summary())
        return solution

    if method not in {"collocation", "scipy"}:
        raise ValueError("method must be 'collocation'/'scipy' or 'pinn'")

    if mesh is None:
        x = create_initial_mesh(problem.domain, n_mesh, kind=mesh_kind)
    else:
        x = validate_mesh(mesh, problem.domain)

    y0 = create_initial_guess(problem, x, guess)
    raw = solve_with_scipy(
        problem,
        x,
        y0,
        tol=tol,
        bc_tol=bc_tol,
        max_nodes=max_nodes,
        verbose=verbose,
    )
    report = build_convergence_report(problem, raw)
    solution = BVPSolution(
        problem=problem,
        x=np.asarray(raw.x),
        y=np.asarray(raw.y),
        diagnostics=report,
        _raw_solution=raw,
        metadata={"backend": "scipy"},
    )
    if raise_on_failure and not solution.success:
        raise RuntimeError(solution.summary())
    return solution

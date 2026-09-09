"""High-level benchmark orchestration."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from pinns4bvp.benchmark.exact import ExactReference
from pinns4bvp.benchmark.metrics import error_metrics
from pinns4bvp.benchmark.report import BenchmarkReport, MethodRun
from pinns4bvp.solver import solve


def _normalize_variables(problem, variables):
    if variables is None:
        return tuple(problem.variable_names)
    names = []
    for variable in variables:
        idx = problem.variable_index(variable)
        name = problem.variable_names[idx]
        if name not in names:
            names.append(name)
    if not names:
        raise ValueError("variables must contain at least one variable")
    return tuple(names)



def _validate_solution(problem, solution, label: str):
    if solution.problem.n_equations != problem.n_equations:
        raise ValueError(f"{label} solution has incompatible n_equations")
    if tuple(solution.problem.domain) != tuple(problem.domain):
        raise ValueError(f"{label} solution has an incompatible domain")
    return solution

def _timed_solve(problem, method: str, kwargs: dict):
    start = perf_counter()
    solution = solve(problem, method=method, **kwargs)
    elapsed = perf_counter() - start
    return MethodRun(method, solution, float(elapsed))


def benchmark_problem(
    problem,
    *,
    exact=None,
    x=None,
    n_points: int = 501,
    variables=None,
    numerical_solution=None,
    pinn_solution=None,
    run_numerical: bool = True,
    run_pinn: bool = True,
    numerical_kwargs: dict | None = None,
    pinn_config=None,
    pinn_kwargs: dict | None = None,
) -> BenchmarkReport:
    """Benchmark numerical collocation, PINN, and an optional exact solution.

    Existing solutions may be supplied to avoid re-solving.  When a solution is
    supplied, its runtime is recorded as ``None`` because the benchmark did not
    perform that solve.
    """

    if n_points < 2:
        raise ValueError("n_points must be at least 2")
    if x is None:
        x = np.linspace(problem.a, problem.b, n_points)
    else:
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or x.size < 2:
            raise ValueError("x must be a one-dimensional grid with at least 2 points")
        if not np.all(np.isfinite(x)):
            raise ValueError("x must contain finite values")
        if np.any(np.diff(x) <= 0):
            raise ValueError("x must be strictly increasing")
        if x[0] < problem.a or x[-1] > problem.b:
            raise ValueError("benchmark grid must lie inside the problem domain")

    variable_names = _normalize_variables(problem, variables)
    exact_ref = None if exact is None else ExactReference(problem, exact)

    numerical_run = None
    if numerical_solution is not None:
        numerical_solution = _validate_solution(problem, numerical_solution, "numerical")
        numerical_run = MethodRun("numerical", numerical_solution, None)
    elif run_numerical:
        numerical_run = _timed_solve(
            problem,
            "collocation",
            dict(numerical_kwargs or {}),
        )

    pinn_run = None
    if pinn_solution is not None:
        pinn_solution = _validate_solution(problem, pinn_solution, "PINN")
        pinn_run = MethodRun("pinn", pinn_solution, None)
    elif run_pinn:
        kwargs = dict(pinn_kwargs or {})
        kwargs["pinn_config"] = pinn_config
        pinn_run = _timed_solve(problem, "pinn", kwargs)

    comparisons: dict[str, dict[str, object]] = {}

    if exact_ref is not None and numerical_run is not None:
        by_variable = {}
        for variable in variable_names:
            if exact_ref.has_variable(variable):
                by_variable[variable] = error_metrics(
                    exact_ref.values(variable, x),
                    numerical_run.solution.values(variable, x),
                )
        if by_variable:
            comparisons["numerical_vs_exact"] = by_variable

    if exact_ref is not None and pinn_run is not None:
        by_variable = {}
        for variable in variable_names:
            if exact_ref.has_variable(variable):
                by_variable[variable] = error_metrics(
                    exact_ref.values(variable, x),
                    pinn_run.solution.values(variable, x),
                )
        if by_variable:
            comparisons["pinn_vs_exact"] = by_variable

    if numerical_run is not None and pinn_run is not None:
        comparisons["pinn_vs_numerical"] = {
            variable: error_metrics(
                numerical_run.solution.values(variable, x),
                pinn_run.solution.values(variable, x),
            )
            for variable in variable_names
        }

    return BenchmarkReport(
        problem=problem,
        x=np.asarray(x, dtype=float),
        variables=variable_names,
        numerical=numerical_run,
        pinn=pinn_run,
        exact=exact_ref,
        comparisons=comparisons,
    )

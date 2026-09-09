import numpy as np
import pytest

from pinns4bvp import BVPProblem, UnknownParameter, solve


def _eq(x, y, p):
    k = p["k"]
    return np.vstack((y[1], -(k**2) * y[0]))


def _bc(ya, yb, p):
    return np.array((ya[0], yb[0], ya[1] - 1.0))


def _guess(x):
    k0 = 3.0
    return np.vstack((np.sin(k0 * x) / k0, np.cos(k0 * x)))


def test_unknown_parameter_validation_and_bc_count():
    p = BVPProblem(
        _eq,
        _bc,
        (0.0, 1.0),
        2,
        unknown_parameters={"k": UnknownParameter(3.0)},
    )
    assert p.n_unknown_parameters == 1
    assert p.n_boundary_residuals == 3
    assert p.unknown_parameter_names == ("k",)
    assert p.parameter_mapping([2.5])["k"] == 2.5

    with pytest.raises(ValueError):
        BVPProblem(
            _eq,
            _bc,
            (0.0, 1.0),
            2,
            parameters={"k": 2.0},
            unknown_parameters={"k": UnknownParameter(3.0)},
        )


def test_collocation_solves_eigenvalue_parameter():
    problem = BVPProblem(
        _eq,
        _bc,
        (0.0, 1.0),
        2,
        unknown_parameters={"k": UnknownParameter(3.0)},
        variable_names=("y", "yp"),
    )
    sol = solve(problem, guess=_guess, tol=1e-9)
    assert sol.success
    assert abs(sol.parameters["k"] - np.pi) < 1e-7
    assert abs(sol.parameter("k") - np.pi) < 1e-7
    assert set(sol.unknown_parameters) == {"k"}


def test_fixed_and_unknown_parameters_share_callback_mapping():
    def eq(x, y, p):
        return np.vstack((y[1], -p["scale"] * p["lambda"] * y[0]))

    def bc(ya, yb, p):
        return np.array((ya[0], yb[0], ya[1] - 1.0))

    problem = BVPProblem(
        eq,
        bc,
        (0.0, 1.0),
        2,
        parameters={"scale": 1.0},
        unknown_parameters={"lambda": UnknownParameter(np.pi**2)},
    )
    mapping = problem.parameter_mapping([2.0])
    assert mapping == {"scale": 1.0, "lambda": 2.0}


def test_benchmark_exact_unknown_parameter():
    from pinns4bvp import benchmark_problem

    problem = BVPProblem(
        _eq,
        _bc,
        (0.0, 1.0),
        2,
        unknown_parameters={"k": UnknownParameter(3.0)},
        variable_names=("y", "yp"),
    )
    report = benchmark_problem(
        problem,
        exact_parameters={"k": np.pi},
        run_pinn=False,
        numerical_kwargs={"guess": _guess, "tol": 1e-9},
    )
    metrics = report.parameter_metrics("numerical_vs_exact", "k")
    assert metrics.abs_error < 1e-7

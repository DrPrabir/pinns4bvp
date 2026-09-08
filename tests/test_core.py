import numpy as np
import pytest

from pinns4bvp import BVPProblem, solve
from pinns4bvp.mesh import create_initial_mesh


def linear_equations(x, y, p):
    return np.vstack((y[1], -np.ones_like(x)))


def linear_bc(ya, yb, p):
    return np.array([ya[0], yb[0]])


def make_linear_problem():
    return BVPProblem(
        equations=linear_equations,
        boundary_conditions=linear_bc,
        domain=(0.0, 1.0),
        n_equations=2,
        variable_names=("y", "yp"),
    )


def test_problem_validation():
    with pytest.raises(ValueError):
        BVPProblem(linear_equations, linear_bc, (1.0, 0.0), 2)


def test_mesh():
    x = create_initial_mesh((0.0, 1.0), 20)
    assert x.shape == (20,)
    assert x[0] == 0.0
    assert x[-1] == 1.0
    assert np.all(np.diff(x) > 0)


def test_linear_bvp_accuracy():
    problem = make_linear_problem()
    sol = solve(problem, tol=1e-8)
    assert sol.success

    x = np.linspace(0.0, 1.0, 101)
    exact = 0.5 * x * (1.0 - x)
    error = np.max(np.abs(sol.values("y", x) - exact))
    assert error < 1e-7
    assert sol.diagnostics.bc_residual_inf < 1e-8


def test_variable_lookup():
    problem = make_linear_problem()
    assert problem.variable_index("yp") == 1
    with pytest.raises(KeyError):
        problem.variable_index("temperature")

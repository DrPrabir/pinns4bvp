import numpy as np
import pytest

from pinns4bvp import BVPProblem, solve
from pinns4bvp.guess import InterpolatedGuess, create_initial_guess, guess_from_solution
from pinns4bvp.mesh import MeshConfig, create_initial_mesh, mesh_quality


def _eq(x, y, p):
    return np.vstack((y[1], -np.ones_like(x)))


def _bc(ya, yb, p):
    return np.array((ya[0], yb[0]))


def _problem():
    return BVPProblem(
        _eq,
        _bc,
        (0.0, 1.0),
        2,
        variable_names=("y", "yp"),
    )


def test_mesh_kinds_and_quality():
    uniform = create_initial_mesh((0.0, 1.0), 21, kind="uniform")
    left = create_initial_mesh((0.0, 1.0), 21, kind="left", power=3.0)
    right = create_initial_mesh((0.0, 1.0), 21, kind="right", power=3.0)
    cheb = create_initial_mesh((0.0, 1.0), 21, kind="chebyshev")

    assert np.allclose(np.diff(uniform), np.diff(uniform)[0])
    assert left[1] - left[0] < left[-1] - left[-2]
    assert right[1] - right[0] > right[-1] - right[-2]
    assert (cheb[1] - cheb[0]) < (cheb[len(cheb)//2] - cheb[len(cheb)//2 - 1])

    report = mesh_quality(left)
    assert report.n_nodes == 21
    assert report.spacing_ratio > 1.0


def test_solver_accepts_mesh_config():
    sol = solve(
        _problem(),
        mesh=MeshConfig(n_nodes=30, kind="chebyshev"),
        tol=1e-8,
    )
    assert sol.success
    quality = sol.metadata["initial_mesh_quality"]
    assert quality.n_nodes == 30


def test_mapping_and_interpolated_guesses():
    problem = _problem()
    x = np.linspace(0.0, 1.0, 25)

    mapping_guess = create_initial_guess(
        problem,
        x,
        {"y": lambda z: 0.5 * z * (1.0 - z), "yp": 0.0},
    )
    assert mapping_guess.shape == (2, 25)
    assert np.max(mapping_guess[0]) > 0
    assert np.allclose(mapping_guess[1], 0.0)

    source_x = np.linspace(0.0, 1.0, 9)
    source_y = np.vstack((source_x, 1.0 - source_x))
    interp = InterpolatedGuess(source_x, source_y)
    y = create_initial_guess(problem, x, interp)
    assert np.allclose(y[0], x)
    assert np.allclose(y[1], 1.0 - x)


def test_previous_solution_can_be_reused_as_guess():
    problem = _problem()
    sol1 = solve(problem, tol=1e-8)
    guess = guess_from_solution(sol1)
    sol2 = solve(problem, n_mesh=37, guess=guess, tol=1e-8)
    assert sol2.success
    xx = np.linspace(0.0, 1.0, 51)
    assert np.max(np.abs(sol1.values("y", xx) - sol2.values("y", xx))) < 1e-8


def test_independent_residual_report_classical():
    sol = solve(_problem(), tol=1e-8)
    report = sol.residual_report(n_points=101)
    assert report.ode_residuals.shape == (2, 101)
    assert report.bc_residuals.shape == (2,)
    assert report.ode_max_abs < 1e-7
    assert report.bc_max_abs < 1e-8
    assert "Residual diagnostics" in report.summary()

import numpy as np
import pytest

from pinns4bvp import (
    BVPProblem,
    ContinuationConfig,
    UnknownParameter,
    continue_parameter,
)


def _linear_eq(x, y, p):
    return np.vstack((y[1], np.zeros_like(x)))


def _linear_bc(ya, yb, p):
    lam = p["lambda"]
    return np.array((ya[0], yb[0] - lam))


def _linear_problem():
    return BVPProblem(
        equations=_linear_eq,
        boundary_conditions=_linear_bc,
        domain=(0.0, 1.0),
        n_equations=2,
        parameters={"lambda": 0.0},
        variable_names=("y", "yp"),
    )


def test_problem_with_parameters_does_not_mutate_original():
    problem = _linear_problem()
    changed = problem.with_parameters(**{"lambda": 2.0})
    assert changed.parameters["lambda"] == 2.0
    assert problem.parameters["lambda"] == 0.0


def test_basic_classical_continuation_and_tracking():
    problem = _linear_problem()
    family = problem.continue_parameter(
        "lambda",
        [0.0, 0.5, 1.0, 1.5],
        solve_kwargs={"tol": 1e-9},
    )
    assert family.success
    assert len(family.solutions) == 4
    assert np.allclose(family.values, [0.0, 0.5, 1.0, 1.5])
    p, response = family.track("y", 0.5)
    assert np.allclose(p, [0.0, 0.5, 1.0, 1.5])
    assert np.allclose(response, 0.5 * p, atol=1e-8)
    assert problem.parameters["lambda"] == 0.0
    assert family.diagnostics.n_requested_converged == 4


def test_continuation_values_must_be_monotonic():
    with pytest.raises(ValueError):
        continue_parameter(_linear_problem(), "lambda", [0.0, 1.0, 0.5])


def test_unknown_parameter_is_reinitialized_from_previous_solution():
    def eq(x, y, p):
        mu = p["mu"]
        k = p["k"]
        return np.vstack((y[1], -(mu * k**2) * y[0]))

    def bc(ya, yb, p):
        return np.array((ya[0], yb[0], ya[1] - 1.0))

    def guess(x):
        return np.vstack((np.sin(np.pi * x) / np.pi, np.cos(np.pi * x)))

    problem = BVPProblem(
        eq,
        bc,
        (0.0, 1.0),
        2,
        parameters={"mu": 1.0},
        unknown_parameters={"k": UnknownParameter(3.0)},
        variable_names=("y", "yp"),
    )
    family = continue_parameter(
        problem,
        "mu",
        [1.0, 2.0],
        initial_guess=guess,
        solve_kwargs={"tol": 1e-9},
    )
    assert family.success
    k1 = family.solution_at(1.0).parameters["k"]
    k2 = family.solution_at(2.0).parameters["k"]
    assert abs(k1 - np.pi) < 1e-7
    assert abs(k2 - np.pi / np.sqrt(2.0)) < 1e-7


def test_adaptive_recovery_inserts_intermediate_points(monkeypatch):
    import pinns4bvp.continuation.parameter as parameter_module

    class DummyProblem:
        def __init__(self, value):
            self.parameters = {"lambda": value}
            self.unknown_parameters = {}
            self.unknown_parameter_names = ()

    class DummySolution:
        def __init__(self, problem, success, source):
            self.problem = problem
            self.success = success
            self.message = "ok" if success else "step too large"
            self.parameters = dict(problem.parameters)
            self.metadata = {"value": problem.parameters["lambda"]}
            self.x = np.array([0.0, 1.0])
            self.y = np.zeros((1, 2))
            self.source = source

    # Use a real BVPProblem so dataclasses.replace in continuation remains tested.
    def eq(x, y, p):
        return np.zeros_like(y)

    def bc(ya, yb, p):
        return np.array((ya[0],))

    problem = BVPProblem(eq, bc, (0.0, 1.0), 1, parameters={"lambda": 0.0})

    def fake_solve(local_problem, *, method="collocation", **kwargs):
        source = kwargs.get("guess") or kwargs.get("pinn_warm_start")
        target = local_problem.parameters["lambda"]
        source_value = 0.0 if source is None else source.metadata["value"]
        ok = source is None or abs(target - source_value) <= 0.30 + 1e-12
        return DummySolution(local_problem, ok, source)

    monkeypatch.setattr(parameter_module, "solve", fake_solve)

    family = parameter_module.continue_parameter(
        problem,
        "lambda",
        [0.0, 1.0],
        config=ContinuationConfig(
            adaptive=True,
            min_step=0.01,
            reduction_factor=0.5,
            max_retries=8,
            record_residuals=False,
        ),
    )
    assert family.success
    assert family.solution_at(1.0).success
    assert any((not point.requested) and point.accepted for point in family.points)
    assert family.diagnostics.n_attempts > 2

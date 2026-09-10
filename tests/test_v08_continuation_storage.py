import numpy as np

from pinns4bvp import Equation, HigherOrderBVP, Parameter, d2, y


def _problem():
    lam = Parameter("lambda", value=0.0)
    problem = HigherOrderBVP(
        equation=Equation(d2(y), 0.0),
        boundary_conditions=[y.at(0.0) == 0.0, y.at(1.0) == lam],
        domain=(0.0, 1.0),
    )
    return problem, lam


def test_parameter_object_and_start_stop_step_continuation():
    problem, lam = _problem()
    family = problem.continue_parameter(
        lam,
        start=0.0,
        stop=1.0,
        step=0.25,
        solve_kwargs={"tol": 1e-9},
    )
    assert family.success
    assert np.allclose(family.values, [0.0, 0.25, 0.5, 0.75, 1.0])
    assert abs(family.solution_at(1.0).values("y", 0.5) - 0.5) < 1e-9


def test_save_all_requested_and_final_policies():
    problem, lam = _problem()

    all_family = problem.continue_parameter(
        lam,
        [0.0, 1.0],
        config=None,
        save="all",
        solve_kwargs={"tol": 1e-9},
    )
    assert all_family.final_solution is not None
    assert all(point.solution is not None for point in all_family.points if point.accepted)

    requested_family = problem.continue_parameter(
        lam,
        [0.0, 0.5, 1.0],
        save="requested",
        solve_kwargs={"tol": 1e-9},
    )
    assert len(requested_family.solutions) == 3

    final_family = problem.continue_parameter(
        lam,
        [0.0, 0.5, 1.0],
        save="final",
        solve_kwargs={"tol": 1e-9},
    )
    assert len(final_family.solutions) == 1
    assert np.allclose(final_family.values, [1.0])
    assert final_family.final_solution is not None
    assert len(final_family.history) == 3
    assert final_family.requested_results[0].solution is None


def test_continue_to_returns_only_target_solution():
    problem, lam = _problem()
    sol = problem.continue_to(
        lam,
        target=1.0,
        start=0.0,
        step=0.2,
        solve_kwargs={"tol": 1e-9},
    )
    assert sol.success
    assert abs(sol.problem.parameters["lambda"] - 1.0) < 1e-14
    assert abs(sol.values("y", 0.5) - 0.5) < 1e-9
    assert "continuation_history" in sol.metadata


def test_high_level_pinn_continuation_reuses_weights():
    import pytest
    pytest.importorskip("torch")
    from pinns4bvp.pinn import PINNConfig

    problem, lam = _problem()
    config = PINNConfig(
        hidden_layers=(8, 8),
        n_collocation=20,
        adam_epochs=5,
        use_lbfgs=False,
        loss_tolerance=100.0,
        early_stopping_patience=None,
        device="cpu",
        dtype="float32",
        verbose=False,
        seed=321,
    )
    family = problem.continue_parameter(
        lam,
        [0.0, 0.25],
        method="pinn",
        pinn_config=config,
        save="requested",
        accept_solution=lambda sol: True,
    )
    assert family.success
    assert family.requested_results[0].solution.metadata["warm_start_used"] is False
    assert family.requested_results[1].solution.metadata["warm_start_used"] is True


def test_start_stop_step_equal_max_step_avoids_float_recursion():
    """Regression: decimal step == max_step must not recurse forever."""
    from pinns4bvp import ContinuationConfig

    problem, lam = _problem()
    family = problem.continue_parameter(
        lam,
        start=0.1,
        stop=0.5,
        step=0.1,
        save="requested",
        solve_kwargs={"tol": 1e-9},
        config=ContinuationConfig(
            adaptive=True,
            min_step=0.00625,
            max_step=0.1,
        ),
    )
    assert family.success
    assert np.allclose(family.requested_values, [0.1, 0.2, 0.3, 0.4, 0.5])
    assert np.allclose(family.values, [0.1, 0.2, 0.3, 0.4, 0.5])


def test_max_step_subdivision_still_makes_progress():
    """A genuinely larger gap should still be subdivided successfully."""
    from pinns4bvp import ContinuationConfig

    problem, lam = _problem()
    family = problem.continue_parameter(
        lam,
        [0.0, 0.3],
        save="all",
        solve_kwargs={"tol": 1e-9},
        config=ContinuationConfig(
            adaptive=True,
            min_step=0.01,
            max_step=0.1,
        ),
    )
    assert family.success
    assert np.allclose(family.requested_values, [0.0, 0.3])
    accepted_values = [p.value for p in family.points if p.accepted]
    assert any(np.isclose(v, 0.1) for v in accepted_values)
    assert any(np.isclose(v, 0.2) for v in accepted_values)
    assert any(np.isclose(v, 0.3) for v in accepted_values)

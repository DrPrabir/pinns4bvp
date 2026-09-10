import numpy as np
import pytest

from pinns4bvp import (
    BVPProblem,
    DependentVariable,
    Equation,
    HigherOrderBVP,
    IndependentVariable,
    Parameter,
    d,
    d2,
    d3,
    exp,
    solve,
    y,
)


def test_default_yx_second_order_problem_matches_exact_solution():
    problem = HigherOrderBVP(
        equation=Equation(d2(y), -1.0),
        boundary_conditions=[y.at(0.0) == 0.0, y.at(1.0) == 0.0],
        domain=(0.0, 1.0),
    )
    sol = problem.solve(tol=1e-10)
    grid = np.linspace(0.0, 1.0, 101)
    exact = 0.5 * grid * (1.0 - grid)
    assert sol.success
    assert np.max(np.abs(sol.values("y", grid) - exact)) < 1e-10
    assert problem.variable_names == ("y", "d(y)/dx")
    assert problem.formulation_metadata.independent_variable == "x"


def test_explicit_dependent_and_independent_variable_names():
    r = IndependentVariable("r")
    T = DependentVariable("T", order=2)
    problem = HigherOrderBVP(
        equation=Equation(d2(T, r), -2.0),
        boundary_conditions=[T.at(0.0) == 0.0, T.at(1.0) == 0.0],
        domain=(0.0, 1.0),
    )
    sol = solve(problem, tol=1e-10)
    grid = np.linspace(0.0, 1.0, 51)
    exact = grid * (1.0 - grid)
    assert sol.success
    assert np.max(np.abs(sol.values("T", grid) - exact)) < 1e-10
    assert problem.formulation_metadata.independent_variable == "r"


def test_nonlinear_coupled_system_with_exact_solution():
    u = DependentVariable("u", order=2)
    v = DependentVariable("v", order=2)
    problem = HigherOrderBVP(
        equations=[
            Equation(d2(u) - 2.0 * u * v, 0.0),
            Equation(d2(v) - 6.0 * u**2 * v, 0.0),
        ],
        boundary_conditions=[
            u.at(0.0) == 1.0,
            u.at(1.0) == 0.5,
            v.at(0.0) == 1.0,
            v.at(1.0) == 0.25,
        ],
        domain=(0.0, 1.0),
    )

    def guess(x):
        return np.vstack((
            1.0 / (1.0 + x),
            -1.0 / (1.0 + x) ** 2,
            1.0 / (1.0 + x) ** 2,
            -2.0 / (1.0 + x) ** 3,
        ))

    sol = solve(problem, guess=guess, tol=1e-9)
    grid = np.linspace(0.0, 1.0, 101)
    u_exact = 1.0 / (1.0 + grid)
    v_exact = 1.0 / (1.0 + grid) ** 2
    assert sol.success
    assert np.max(np.abs(sol.values("u", grid) - u_exact)) < 1e-8
    assert np.max(np.abs(sol.values("v", grid) - v_exact)) < 1e-8


def test_mixed_order_system_and_coupled_highest_derivatives():
    u = DependentVariable("u", order=3)
    v = DependentVariable("v", order=2)
    problem = HigherOrderBVP(
        equations=[
            Equation(d3(u) + d2(v), 8.0),
            Equation(d3(u) - d2(v), 4.0),
        ],
        boundary_conditions=[
            u.at(0.0) == 0.0,
            d(u).at(0.0) == 0.0,
            u.at(1.0) == 1.0,
            v.at(0.0) == 0.0,
            v.at(1.0) == 1.0,
        ],
        domain=(0.0, 1.0),
    )

    def guess(x):
        return np.vstack((x**3, 3 * x**2, 6 * x, x**2, 2 * x))

    sol = solve(problem, guess=guess, tol=1e-10)
    grid = np.linspace(0.0, 1.0, 51)
    assert sol.success
    assert np.max(np.abs(sol.values("u", grid) - grid**3)) < 1e-10
    assert np.max(np.abs(sol.values("v", grid) - grid**2)) < 1e-10
    assert len(problem.variable_names) == 5


def test_high_level_unknown_parameter_eigenvalue():
    k = Parameter("k", initial=3.0, unknown=True)
    problem = HigherOrderBVP(
        equation=Equation(d2(y) + k**2 * y, 0.0),
        boundary_conditions=[
            y.at(0.0) == 0.0,
            y.at(1.0) == 0.0,
            d(y).at(0.0) == 1.0,
        ],
        domain=(0.0, 1.0),
    )

    def guess(x):
        return np.vstack((np.sin(np.pi * x) / np.pi, np.cos(np.pi * x)))

    sol = solve(problem, guess=guess, tol=1e-10)
    assert sol.success
    assert abs(sol.parameter("k") - np.pi) < 1e-9


def test_high_level_equation_generates_pinn_callbacks():
    torch = pytest.importorskip("torch")
    a = Parameter("a", value=2.0)
    problem = HigherOrderBVP(
        equation=Equation(d2(y) + a * y, 0.0),
        boundary_conditions=[y.at(0.0) == 0.0, y.at(1.0) == 1.0],
        domain=(0.0, 1.0),
    )
    xt = torch.linspace(0, 1, 8)
    state = torch.stack((xt, torch.ones_like(xt)))
    rhs = problem.evaluate_pinn_equations(xt, state)
    assert tuple(rhs.shape) == tuple(state.shape)
    bc = problem.evaluate_pinn_boundary_conditions(state[:, 0], state[:, -1])
    assert bc.numel() == 2


def test_bratu_style_exp_expression_compiles():
    lam = Parameter("lambda", value=1.0)
    problem = HigherOrderBVP(
        equation=Equation(d2(y) + lam * exp(y), 0.0),
        boundary_conditions=[y.at(0.0) == 0.0, y.at(1.0) == 0.0],
        domain=(0.0, 1.0),
    )
    sol = solve(problem, guess="zeros", tol=1e-6)
    assert sol.success


def test_invalid_boundary_condition_count_is_caught_early():
    with pytest.raises(ValueError, match="incorrect number of boundary conditions"):
        HigherOrderBVP(
            equation=Equation(d2(y), -1.0),
            boundary_conditions=[y.at(0.0) == 0.0],
            domain=(0.0, 1.0),
        )


def test_nonlinear_highest_derivative_is_rejected():
    with pytest.raises(ValueError, match="highest derivatives"):
        HigherOrderBVP(
            equation=Equation(d2(y) ** 2 + y, 0.0),
            boundary_conditions=[y.at(0.0) == 0.0, y.at(1.0) == 0.0],
            domain=(0.0, 1.0),
        )


def test_boundary_conditions_must_be_at_endpoints():
    with pytest.raises(ValueError, match="domain endpoints"):
        HigherOrderBVP(
            equation=Equation(d2(y), -1.0),
            boundary_conditions=[y.at(0.25) == 0.0, y.at(1.0) == 0.0],
            domain=(0.0, 1.0),
        )


def test_high_level_problem_can_run_through_pinn_backend():
    pytest.importorskip("torch")
    from pinns4bvp.pinn import PINNConfig

    problem = HigherOrderBVP(
        equation=Equation(d2(y), -1.0),
        boundary_conditions=[y.at(0.0) == 0.0, y.at(1.0) == 0.0],
        domain=(0.0, 1.0),
    )
    config = PINNConfig(
        hidden_layers=(8, 8),
        n_collocation=24,
        adam_epochs=10,
        use_lbfgs=False,
        loss_tolerance=10.0,
        early_stopping_patience=None,
        device="cpu",
        dtype="float32",
        verbose=False,
        seed=123,
    )
    sol = problem.solve(method="pinn", pinn_config=config)
    assert sol.metadata["backend"] == "pinn"
    assert sol.metadata["device_resolved"] == "cpu"
    assert sol.y.shape[0] == 2


def test_pinn_backend_functions_accept_fixed_scalar_parameters():
    torch = pytest.importorskip("torch")
    from pinns4bvp import IndependentVariable, DependentVariable

    Y = IndependentVariable("y")
    u = DependentVariable("u", order=2)
    K = Parameter("K", value=5.0)

    # Regression for Figure 10-style cosh(K*y)/cosh(K), written via exp.
    cosh_KY = 0.5 * (exp(K * Y) + exp(-K * Y))
    cosh_K = 0.5 * (exp(K) + exp(-K))
    problem = HigherOrderBVP(
        equation=Equation(d2(u, Y) + cosh_KY / cosh_K, 0.0),
        boundary_conditions=[u.at(-1.0) == 0.0, u.at(1.0) == 0.0],
        domain=(-1.0, 1.0),
    )

    coord = torch.linspace(-1.0, 1.0, 9)
    state = torch.stack((torch.zeros_like(coord), torch.zeros_like(coord)))
    rhs = problem.evaluate_pinn_equations(coord, state)
    assert tuple(rhs.shape) == tuple(state.shape)
    assert torch.isfinite(rhs).all()

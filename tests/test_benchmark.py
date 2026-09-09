import numpy as np
import pytest

from pinns4bvp import BVPProblem, benchmark_problem, solve
from pinns4bvp.benchmark import ExactReference, error_metrics


def _np_eq(x, y, p):
    return np.vstack((y[1], -np.ones_like(x)))


def _np_bc(ya, yb, p):
    return np.array([ya[0], yb[0]])


def _problem(**kwargs):
    return BVPProblem(
        _np_eq,
        _np_bc,
        (0.0, 1.0),
        2,
        variable_names=("y", "yp"),
        name="Benchmark test",
        **kwargs,
    )


def _exact(x):
    x = np.asarray(x, dtype=float)
    return np.vstack((0.5 * x * (1.0 - x), 0.5 - x))


def test_error_metrics_exact_match_and_zero_reference():
    ref = np.zeros(5)
    same = error_metrics(ref, ref)
    assert same.rmse == 0.0
    assert same.relative_l2 == 0.0

    different = error_metrics(ref, np.ones(5))
    assert np.isinf(different.relative_l2)
    assert different.mae == 1.0


def test_exact_reference_callable_and_mapping():
    problem = _problem()
    x = np.linspace(0.0, 1.0, 11)

    full = ExactReference(problem, _exact)
    assert np.allclose(full.values("y", x), 0.5 * x * (1.0 - x))
    assert np.allclose(full.values("yp", x), 0.5 - x)

    mapped = ExactReference(
        problem,
        {
            "y": lambda z: 0.5 * z * (1.0 - z),
            "yp": lambda z: 0.5 - z,
        },
    )
    assert np.allclose(mapped.values("y", x), full.values("y", x))


def test_numerical_vs_exact_benchmark():
    problem = _problem()
    report = benchmark_problem(
        problem,
        exact=_exact,
        run_pinn=False,
        numerical_kwargs={"tol": 1e-10},
        n_points=101,
    )
    assert report.numerical is not None
    assert report.pinn is None
    assert report.numerical.runtime_seconds is not None
    assert report.metrics("numerical_vs_exact", "y").max_abs_error < 1e-10
    assert report.metrics("numerical_vs_exact", "yp").max_abs_error < 1e-10
    assert "Numerical Vs Exact" in report.summary()


def test_precomputed_solutions_do_not_claim_runtime():
    problem = _problem()
    numerical = solve(problem, method="collocation", tol=1e-10)
    report = benchmark_problem(
        problem,
        exact=_exact,
        numerical_solution=numerical,
        run_pinn=False,
    )
    assert report.numerical.runtime_seconds is None
    data = report.to_dict()
    assert data["runtime_seconds"]["numerical"] is None


def test_full_numerical_pinn_exact_benchmark():
    torch = pytest.importorskip("torch")
    from pinns4bvp.pinn import PINNConfig

    def torch_eq(x, y, p):
        return torch.stack((y[1], -torch.ones_like(x)))

    def torch_bc(ya, yb, p):
        return torch.stack((ya[0], yb[0]))

    problem = _problem(
        pinn_equations=torch_eq,
        pinn_boundary_conditions=torch_bc,
    )
    config = PINNConfig(
        hidden_layers=(16, 16),
        n_collocation=48,
        adam_epochs=600,
        adam_lr=2e-3,
        use_lbfgs=True,
        lbfgs_max_iter=80,
        bc_weight=10.0,
        seed=321,
        history_every=50,
        verbose=False,
        early_stopping_patience=None,
    )
    report = benchmark_problem(
        problem,
        exact=_exact,
        n_points=101,
        numerical_kwargs={"tol": 1e-10},
        pinn_config=config,
    )

    assert report.numerical is not None
    assert report.pinn is not None
    assert report.exact is not None
    assert set(report.comparisons) == {
        "numerical_vs_exact",
        "pinn_vs_exact",
        "pinn_vs_numerical",
    }
    assert report.metrics("pinn_vs_exact", "y").max_abs_error < 2e-3
    assert report.metrics("pinn_vs_numerical", "y").max_abs_error < 2e-3

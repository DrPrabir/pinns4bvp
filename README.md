# PINNs4BVP

PINNs4BVP is an alpha-stage, general-purpose Python framework for two-point boundary-value problems (BVPs). It provides a common problem/solution interface for classical collocation and physics-informed neural-network (PINN) backends.

## v0.4 development focus

The v0.4 development snapshot adds a general benchmarking layer for comparing:

- classical numerical collocation,
- PyTorch PINN solutions,
- exact/analytical solutions when available.

The benchmark engine evaluates all methods on a common grid and reports RMSE, MAE, maximum absolute error, absolute L2 error, relative L2 error, solver/training success, and measured runtimes for solves performed by the benchmark.

## Installation for development

```bash
python -m pip install -e ".[dev]"
```

For users who only need the classical backend:

```bash
python -m pip install -e .
```

## Benchmark example

```python
from pinns4bvp import benchmark_problem

report = benchmark_problem(
    problem,
    exact=exact_solution,
    numerical_kwargs={"tol": 1e-10},
    pinn_config=pinn_config,
)

print(report.summary())
report.plot("y")
```

An exact solution may be supplied either as a callable returning the complete first-order state or as a mapping from variable names to exact callables.

See:

```bash
python -m pinns4bvp.examples.benchmark_linear_bvp
```

## Development status

v0.4 is a development snapshot, not a stable release. The public API may change before v1.0. Numerical and PINN results should be independently verified for research use.

## License

MIT. See `LICENSE` and `THIRD_PARTY.md`.

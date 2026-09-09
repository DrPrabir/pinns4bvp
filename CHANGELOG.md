# Changelog

## 0.4.0.dev0

### Added

- General `benchmark_problem(...)` runner.
- Numerical vs exact comparison.
- PINN vs exact comparison.
- PINN vs numerical comparison.
- Common-grid evaluation for all benchmark methods.
- RMSE, MAE, maximum absolute error, absolute L2, and relative L2 metrics.
- Exact reference adapter supporting full-state callables and variable mappings.
- Structured `BenchmarkReport` and `MethodRun` objects.
- Benchmark runtime measurement for solves executed by the runner.
- Benchmark dictionary export through `BenchmarkReport.to_dict()`.
- Numerical/PINN/exact overlay plotting.
- General linear BVP benchmark example and benchmark tests.

### Changed

- Refactored the previous single `benchmark.py` module into the `pinns4bvp.benchmark` package while preserving `compare_solutions` imports.
- Top-level package now exports `benchmark_problem`.

### Scientific note

A PINN runtime reported by `benchmark_problem` includes the PINN solve/training call. A numerical runtime includes the complete collocation solve call. If a precomputed solution is supplied, runtime is reported as not measured rather than as zero.

## 0.3.0.dev0

- Robust PINN training development snapshot with Adam, optional L-BFGS, reproducibility controls, training diagnostics, stopping criteria, and gradient diagnostics.

## 0.2.0.dev0

- Initial PyTorch PINN development structure.

## 0.1.0

- Initial classical BVP framework using SciPy `solve_bvp` as the numerical backend.

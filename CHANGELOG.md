# Changelog

## Unreleased - v0.2 development

### Scaffold added
- Optional PyTorch dependency group: `pinn`.
- `pinn/config.py` for PINN configuration.
- `pinn/network.py` reserved for the fully connected network.
- `pinn/autodiff.py` reserved for PyTorch derivatives.
- `backends/pinn_backend.py` reserved for PINN solving.
- `diagnostics/pinn_training.py` for training-history storage.
- `benchmark.py` reserved for backend comparison metrics.
- `examples/pinn_linear_bvp.py` reserved for the first PINN validation case.
- Scaffold tests that preserve the existing v0.1 behavior.

## 0.1.0 - 2026-09-09

Initial alpha release.

### Added
- `BVPProblem` for defining nonlinear two-point boundary-value problems.
- `solve()` public solver interface.
- `BVPSolution` wrapper independent of the SciPy result object.
- SciPy `solve_bvp` backend.
- Initial mesh utilities.
- Initial-guess utilities.
- Convergence diagnostics.
- Linear BVP, Bratu, and Blasius examples.
- Basic automated tests.

### Planned
- Parameter continuation.
- Semi-infinite-domain/far-field assistance.
- Research-oriented wall quantities and tabulation.
- Multiple-solution/branch utilities.
- PINN backend and benchmarking.

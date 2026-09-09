# Changelog

## 0.5.0.dev0

### Added

- `UnknownParameter` / `Unknown` public parameter specification.
- `BVPProblem.unknown_parameters` for scalar parameters solved with the BVP state.
- Combined fixed/unknown callback parameter mappings.
- Native SciPy `solve_bvp(..., p=...)` integration for unknown parameters.
- Joint PINN optimization of network weights and unknown scalar parameters.
- Unknown-parameter values in PINN training history.
- `BVPSolution.parameters`, `BVPSolution.unknown_parameters`, and `parameter(name)`.
- Parameter reference/error metrics in the benchmarking framework.
- Classical eigenvalue BVP example recovering `k = pi`.
- PINN eigenvalue BVP example.
- Unknown-parameter and eigenvalue regression tests.

### Changed

- Boundary callbacks now return `n_equations + n_unknown_parameters` residuals when unknown parameters are declared.
- Convergence diagnostics evaluate boundary residuals with the solved SciPy parameter vector.
- PINN optimizer includes unknown scalar parameters in both Adam and L-BFGS stages.

### Compatibility

Problems without unknown parameters retain the v0.4 callback and solver behavior.

## 0.4.0.dev0

- General numerical vs PINN vs exact benchmarking framework.
- Standard error metrics, timing, benchmark reports, and plots.

## 0.3.0.dev0

- Robust Adam + L-BFGS PINN training.
- Diagnostics and reproducibility controls.

## 0.2.0.dev0

- Initial PyTorch PINN backend development.

## 0.1.0

- Basic general BVP framework using the SciPy collocation backend.

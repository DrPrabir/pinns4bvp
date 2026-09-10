# Changelog

## 0.7.0.dev0

### Added

- `ContinuationConfig` for natural parameter continuation and adaptive recovery.
- `continue_parameter(...)` public API.
- `BVPProblem.continue_parameter(...)` convenience wrapper.
- `ContinuationPoint`, `SolutionFamily`, and `ContinuationDiagnostics`.
- Automatic reuse of previous classical solutions as initial guesses.
- Optional `max_step` subdivision and failed-step interval reduction.
- Continuation with simultaneous v0.5 unknown-parameter/eigenvalue solving.
- Family tracking, evaluation, profile plotting, and tracked-response plotting.
- Experimental PINN model warm starts between continuation points.
- Warm starts for solved unknown PINN parameters.
- Linear, Bratu, eigenvalue-continuation, and PINN-continuation examples.
- v0.7 continuation and PINN warm-start regression tests.

### Changed

- `solve(...)` accepts `pinn_warm_start` for compatible previous PINN solutions.
- PINN training accepts an initial model state and optional unknown-parameter initial values.
- `BVPProblem.with_parameters(...)` returns a non-mutating parameter-updated problem copy.
- Package version advanced to `0.7.0.dev0`.

### Compatibility

- v0.6 mesh, guess, residual-diagnostic, and CPU/CUDA/MPS APIs are retained.
- v0.5 fixed/unknown parameter APIs are retained.
- Classical `solve(...)` and existing PINN calls remain backward compatible.

## 0.6.0.dev0

### Added

- Better initial meshes and mesh diagnostics.
- Reusable/interpolated initial guesses.
- Independent residual diagnostics.
- CPU, CUDA, MPS, and automatic PINN device selection.

## 0.5.0.dev0

### Added

- First-class scalar unknown parameters.
- Classical unknown-parameter/eigenvalue solving.
- PINN trainable unknown parameters.
- Exact-parameter benchmarking and parameter histories.

## 0.4.0.dev0

### Added

- General numerical/PINN/exact benchmarking framework.

## 0.3.0.dev0

### Added

- Robust Adam + L-BFGS PINN training, diagnostics, and reproducibility controls.

## 0.2.0.dev0

### Added

- Initial PyTorch PINN backend.

## 0.1.0

### Added

- Basic general BVP framework using the SciPy collocation backend.

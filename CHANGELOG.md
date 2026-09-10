# Changelog

## 0.6.0.dev0

### Added

- `MeshConfig` for reusable initial-mesh configuration.
- Uniform, left-clustered, right-clustered, and Chebyshev/cosine-clustered meshes.
- `MeshQualityReport` and `mesh_quality(...)` spacing diagnostics.
- `InterpolatedGuess` and `guess_from_solution(...)`.
- Initial guesses from previous solutions, source `(x, y)` data, and variable mappings.
- Backend-independent `ResidualReport` and `BVPSolution.residual_report()`.
- `BVPSolution.plot_residuals()`.
- PINN device discovery and selection for CPU, CUDA, MPS, and automatic selection.
- Requested/resolved device metadata on PINN solutions.
- Device-selection and residual-diagnostics examples.
- v0.6 regression tests for mesh, guess, residual, and device behavior.

### Changed

- `solve(...)` now accepts `MeshConfig` and a `mesh_power` option while preserving legacy mesh arguments.
- `PINNConfig.device` is now validated against `auto`, `cpu`, `cuda`, and `mps`.
- PINN training resolves device availability before moving models and trainable unknown parameters.
- `auto` preserves requested float64 precision by avoiding MPS when only float32 MPS execution is allowed.

### Compatibility

- v0.5 fixed/unknown parameter APIs are retained.
- Existing `mesh_kind="quadratic"` remains supported as a left-clustered mesh alias.
- Existing arrays, callable guesses, and `guess="zeros"` remain supported.

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

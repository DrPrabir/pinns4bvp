# Changelog

## 0.8.0.dev1

### Fixed

- Fixed an infinite-recursion bug in parameter continuation when a decimal `step` (for example `0.1`) was mathematically equal to `ContinuationConfig.max_step` but differed by floating-point roundoff.
- Added machine-precision-tolerant max-step comparison and no-progress guards for proactive continuation subdivision.
- Added regression tests for `start=0.1, stop=0.5, step=0.1, max_step=0.1` and for genuine max-step subdivision.

## 0.8.0.dev0

### Added

- Higher-order mathematical formulation package `pinns4bvp.formulation`.
- Default `x` and `y` symbols for ordinary `y(x)` problems.
- `IndependentVariable`, `DependentVariable`, `Parameter`, `Equation`, and `HigherOrderBVP` APIs.
- `d`, `d2`, `d3`, `d4`, and general `derivative(..., order=n)` helpers.
- Backend-neutral `exp`, `sin`, `cos`, `tanh`, `sqrt`, and `log` expression functions.
- Automatic first-order conversion for scalar higher-order equations.
- Automatic first-order conversion for simultaneous and mixed-order systems.
- Linear coupled highest-derivative system handling.
- Endpoint boundary-condition notation with `.at(...)`.
- High-level known and unknown/eigen parameters.
- Automatic boundary-condition count and formulation validation.
- `BVPSolution.derivative(...)` convenience method.
- Continuation `start`/`stop`/`step` sequence generation.
- Continuation storage policies `all`, `requested`, and `final`.
- Lightweight `SolutionFamily.history` under every storage policy.
- `SolutionFamily.final_solution` and `BVPProblem.continue_to(...)`.
- Optional low-level `equations_jacobian`, `boundary_jacobian`, and `singular_matrix` hooks for SciPy collocation.
- Higher-order linear/nonlinear, simultaneous, mixed-order, eigenvalue, continuation, and PINN examples.
- v0.8 formulation, retention-policy, Jacobian, and validation regression tests.

### Changed

- `BVPProblem` gains a `solve(...)` convenience method.
- `BVPProblem.continue_parameter(...)` accepts high-level named `Parameter` objects and optional generated ranges.
- `continue_parameter(...)` accepts either explicit `values` or `start`/`stop`/`step`.
- Package version advanced to `0.8.0.dev0`.

### Compatibility

- The low-level first-order `BVPProblem` interface remains supported.
- Existing separate NumPy/PyTorch callbacks remain supported.
- v0.5 unknown parameters/eigenvalue APIs remain supported.
- v0.6 mesh, reusable guess, residual-diagnostic, and CPU/CUDA/MPS APIs remain supported.
- v0.7 natural continuation, adaptive recovery, `SolutionFamily`, and PINN warm-start behavior remain supported.
- `save="all"` is the default continuation retention policy so v0.7 intermediate-solution behavior is preserved.

## 0.7.0.dev0

### Added

- Natural one-parameter continuation, adaptive recovery, `SolutionFamily`, and PINN warm starts.

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

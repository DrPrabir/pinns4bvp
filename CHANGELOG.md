# Changelog

## v0.3.0.dev0 — development

### Added
- Reproducible PyTorch PINN training with explicit random seeds.
- Domain-normalized fully connected neural network.
- Automatic differentiation for first-order BVP state systems.
- Weighted ODE and boundary-condition losses.
- Adam warm-up followed by optional L-BFGS refinement.
- Early stopping, loss tolerance, optional gradient clipping, and gradient-norm diagnostics.
- Structured PINN training history and stop reasons.
- Backend-independent `BVPSolution` evaluator.
- `solve(..., method="pinn")` alongside the existing collocation backend.
- General solution-comparison metrics.
- PINN linear-BVP example and core PINN tests.

### Development status
This is a development snapshot, not a tagged stable release.  The public PINN
callback API may still change before v1.0.

## v0.1.0
- Initial classical BVP framework using SciPy `solve_bvp`.

# PINNs4BVP v0.3 training development notes

This snapshot is intended for development on a feature branch.  It is not a
stable release tag.

## Training pipeline

1. Set Python / NumPy / PyTorch seeds before network construction.
2. Build a domain-normalized MLP.
3. Construct deterministic uniform or seeded-random collocation points.
4. Minimize weighted first-order ODE and boundary residual losses with Adam.
5. Optionally refine with PyTorch L-BFGS.
6. Record total, ODE, BC, and gradient-norm diagnostics.
7. Return a backend-independent `BVPSolution` with training metadata.

## Current alpha API limitation

The classical backend accepts NumPy callbacks.  The PINN backend currently
requires equivalent PyTorch-native callbacks (`pinn_equations` and
`pinn_boundary_conditions`) so autograd is not broken by NumPy conversion.
A later API revision should reduce this duplication.

## Validation performed

- Existing linear, Bratu, and Blasius collocation examples still converge.
- Blasius gives f''(0) = 0.33205734.
- PINN network/autodiff/reproducibility tests pass.
- A linear BVP PINN is checked against the exact/collocation solution.

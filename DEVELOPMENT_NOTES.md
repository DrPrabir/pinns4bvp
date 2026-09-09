# PINNs4BVP v0.5 development notes

## Scope

v0.5 introduces **unknown scalar parameters and eigenvalue BVPs** while preserving the general-purpose first-order BVP abstraction.

The mathematical interface is

`y' = f(x, y, p)`

with boundary residuals

`g(ya, yb, p) = 0`.

For `n` state equations and `k` unknown parameters, `g` must return `n + k` residuals.

## Design decisions

1. Fixed and unknown parameters are declared separately.
2. User callbacks receive a single combined mapping.
3. The SciPy backend uses SciPy's native unknown-parameter vector rather than an outer root-search loop.
4. The PINN backend represents each unknown scalar as a trainable `torch.nn.Parameter` and optimizes it jointly with network weights.
5. Solved parameter values are backend-independent and exposed on `BVPSolution.parameters`.
6. Benchmarking can compare estimated parameters with known references.

## Deliberate limitations

- Unknowns are unconstrained scalar parameters only.
- No bounds or positive/log transforms yet.
- No observation/data loss: v0.5 is not yet a full inverse-problem framework.
- No automatic eigenmode indexing or branch discovery.
- PINN and NumPy callbacks remain separate during the alpha API period.

## Validation target

The canonical test problem is

`y'' + k^2 y = 0`, `y(0)=0`, `y(1)=0`, `y'(0)=1`.

The first positive mode has `k = pi`. The classical backend should recover this value to high accuracy; the PINN backend should learn it within a practical optimization tolerance.

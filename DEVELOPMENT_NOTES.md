# PINNs4BVP v0.4 development notes

## Scope

v0.4 is intentionally application-neutral. The benchmark layer works with any `BVPProblem` that can be evaluated through the existing numerical and/or PINN backend.

## Exact-reference convention

An exact reference can be supplied in either form:

```python
def exact(x):
    return np.vstack((y_exact(x), yp_exact(x)))
```

or:

```python
exact = {
    "y": y_exact,
    "yp": yp_exact,
}
```

This avoids embedding application-specific analytical solutions inside `BVPProblem`.

## Runtime interpretation

Timing comparisons must be interpreted carefully. Classical collocation time and PINN training time represent different computational workflows. v0.4 reports them transparently but does not claim that they are directly equivalent measures of efficiency. PINN inference timing is not yet reported separately.

## Planned follow-up

Before a stable v0.4 release, consider adding:

- optional JSON/CSV export helpers,
- separate PINN training and inference timing,
- residual-norm benchmarking,
- configurable benchmark grids,
- multi-run PINN statistics for stochastic robustness.

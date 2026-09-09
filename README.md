# PINNs4BVP v0.5 development snapshot

PINNs4BVP is a general-purpose Python framework for two-point boundary-value problems with classical collocation and physics-informed neural-network backends.

**v0.5 adds unknown scalar parameters and eigenvalue BVPs.** Unknown parameters are solved simultaneously with the state rather than through an external parameter-search loop.

> Status: alpha development snapshot. This is not yet a stable release.

## Installation for development

```bash
python -m pip install -e ".[dev]"
```

Classical-only users can install the base dependencies with:

```bash
python -m pip install -e .
```

## Fixed versus unknown parameters

Fixed parameters remain ordinary scalars:

```python
parameters={"alpha": 2.0}
```

Unknown parameters are declared separately:

```python
from pinns4bvp import UnknownParameter

unknown_parameters={"k": UnknownParameter(initial=3.0)}
```

Callbacks receive one combined parameter mapping, so both fixed and unknown values are accessed in the same way:

```python
def equations(x, y, p):
    k = p["k"]
    return ...
```

For `n` state equations and `k` unknown parameters, the boundary callback must return exactly `n + k` residuals.

## Eigenvalue example

Solve

\[
y'' + k^2 y = 0, \qquad 0 \le x \le 1,
\]

with

\[
y(0)=0, \qquad y(1)=0, \qquad y'(0)=1.
\]

The normalization removes the trivial solution and the first positive eigenvalue is `k = pi`.

```python
import numpy as np
from pinns4bvp import BVPProblem, UnknownParameter, solve


def fun(x, y, p):
    k = p["k"]
    return np.vstack((y[1], -(k**2) * y[0]))


def bc(ya, yb, p):
    return np.array((ya[0], yb[0], ya[1] - 1.0))


def guess(x):
    k0 = 3.0
    return np.vstack((np.sin(k0*x)/k0, np.cos(k0*x)))


problem = BVPProblem(
    equations=fun,
    boundary_conditions=bc,
    domain=(0.0, 1.0),
    n_equations=2,
    unknown_parameters={"k": UnknownParameter(3.0)},
    variable_names=("y", "yp"),
)

sol = solve(problem, guess=guess, tol=1e-9)
print(sol.parameters["k"])
```

The bundled collocation example recovers `k` approximately equal to `pi`.

Run it with:

```bash
python -m pinns4bvp.examples.eigenvalue_bvp
```

## PINN unknown-parameter solving

The PINN backend jointly optimizes neural-network weights and unknown scalar parameters. PyTorch-native equation and boundary callbacks are still required in the alpha API.

```bash
python -m pinns4bvp.examples.pinn_eigenvalue_bvp
```

Training history records the evolution of unknown parameters as well as ODE loss, BC loss, and gradient norms.

## Accessing solved parameters

```python
sol.parameters
sol.parameters["k"]
sol.parameter("k")
sol.unknown_parameters
```

Fixed parameters and solved unknown parameters are both preserved in `sol.parameters`.

## Benchmarking unknown parameters

v0.5 extends the v0.4 benchmark framework with exact parameter references:

```python
report = benchmark_problem(
    problem,
    exact_parameters={"k": np.pi},
    run_pinn=False,
    numerical_kwargs={"guess": guess, "tol": 1e-9},
)

print(report.summary())
```

Parameter reports include the reference value, estimated value, absolute error, and relative error.

## Current limitations

- Unknown parameters are scalar and unconstrained in v0.5.
- Parameter bounds/transforms are not yet implemented.
- This is not yet a general data-driven inverse-problem API.
- PINN callbacks must currently be provided separately from NumPy callbacks.
- Eigenvalue branch selection depends on the initial state/parameter guess, as expected for nonlinear BVP solvers.

## Tests

```bash
pytest -v
```

The development snapshot includes tests for classical unknown-parameter solving, PINN parameter learning, parameter diagnostics, and exact-parameter benchmarking.

## License

MIT. See `LICENSE` and `THIRD_PARTY.md`.

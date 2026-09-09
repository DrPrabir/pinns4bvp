# PINNs4BVP v0.1

**PINNs4BVP** is an early-stage, researcher-friendly Python framework for
nonlinear two-point boundary-value problems. Its long-term goal is to provide a
MATLAB `bvp4c`-like workflow for boundary-layer, unidirectional-flow,
heat/mass-transfer, MHD, and related nonlinear ODE systems, while later adding
physics-informed neural networks as an alternative solver and benchmarking
backend.

> **Important:** v0.1 uses `scipy.integrate.solve_bvp` as its numerical backend.
> A PINN backend is planned, but is not included in v0.1.

## v0.1 scope

- Define a first-order BVP using `BVPProblem`
- Fixed physical parameters supplied as a dictionary
- Uniform or left-clustered quadratic initial mesh
- Zero, array, or callable initial guesses
- SciPy collocation backend
- Backend-independent `BVPSolution`
- Convergence diagnostics
- Linear, Bratu, and Blasius examples
- Basic automated tests

Not yet included: unknown/eigenparameters, automatic semi-infinite-domain
extension, continuation, branch tracking, symbolic higher-order equation
parsing, or PINNs.

## Install locally

From the repository root:

```bash
python -m pip install -e .
```

For testing:

```bash
python -m pip install -e ".[test]"
pytest
```

## Install directly from GitHub

```bash
python -m pip install git+https://github.com/DrPrabir/pinns4bvp.git
```

## Minimal example

```python
import numpy as np
from pinns4bvp import BVPProblem, solve


def equations(x, y, p):
    return np.vstack((y[1], -np.ones_like(x)))


def bc(ya, yb, p):
    return np.array([ya[0], yb[0]])


problem = BVPProblem(
    equations=equations,
    boundary_conditions=bc,
    domain=(0.0, 1.0),
    n_equations=2,
    variable_names=("y", "y_prime"),
)

sol = solve(problem, tol=1e-8)
print(sol.summary())
print(sol.values("y", 0.5))
```

## Run examples

```bash
python -m pinns4bvp.examples.linear_bvp
python -m pinns4bvp.examples.bratu
python -m pinns4bvp.examples.blasius
```

The Blasius example should recover the standard wall-shear benchmark
`f''(0) ≈ 0.332057` for the chosen far-field truncation and tolerance.

## API philosophy

The public workflow is intentionally small:

```text
BVPProblem  ->  solve(...)  ->  BVPSolution
```

The numerical backend is kept separate so future releases can add parameter
continuation, custom collocation methods, and PINN backends without requiring
researchers to rewrite their problem definitions.

## License and third-party software

PINNs4BVP is released under the MIT License and is primarily intended for
education and academic research. The MIT License itself does not impose an
"educational use only" restriction.

See `THIRD_PARTY.md` for the open-source libraries used by the project and their
licenses.

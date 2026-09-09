# PINNs4BVP

**PINNs4BVP** is a general-purpose Python framework for solving two-point boundary-value problems (BVPs) using classical numerical methods and, in ongoing development, physics-informed neural networks (PINNs).

The project is designed to provide a simple research-oriented interface for defining, solving, validating, and eventually benchmarking BVPs through interchangeable numerical and PINN backends.

> **Project status:** pre-1.0 research software.  
> The `main` branch currently represents the stable classical BVP baseline. New capabilities are developed and tested on versioned development branches before being merged into `main`.

---

## Project status

| Version / branch | Status | Main capability |
|---|---|---|
| `v0.1.0` | Stable tagged baseline | General BVP framework using SciPy `solve_bvp` |
| `v0.2-pinn-backend` | Development | Initial PyTorch PINN backend |
| `v0.3-robust-training` | Development | Adam + L-BFGS, diagnostics, reproducibility |
| `v0.4-general-benchmarking` | Development | Numerical vs PINN vs exact benchmarking |
| `v0.5-unknown-parameters` | Development | Unknown parameters and eigenvalue BVPs |

The stable tagged release is currently:

```text
v0.1.0
```

---

## Scope

PINNs4BVP is intended as a **general-purpose ODE boundary-value problem framework**.

The core mathematical form is

\[
\mathbf{y}'(x) = \mathbf{f}(x,\mathbf{y},p),
\]

subject to boundary conditions of the form

\[
\mathbf{g}(\mathbf{y}(a),\mathbf{y}(b),p)=0.
\]

The framework is intended to support a broad range of BVPs arising in applied mathematics, physics, engineering, mechanics, chemistry, biology, eigenvalue analysis, and related fields.

The project is **not limited to boundary-layer or flow problems**.

---

## Current stable capability

The `main` branch and `v0.1.0` tag currently provide:

- a general `BVPProblem` interface;
- a SciPy `solve_bvp` backend;
- user-defined systems of first-order ODEs;
- user-defined two-point boundary conditions;
- configurable initial meshes;
- configurable initial guesses;
- a backend-independent `BVPSolution` wrapper;
- convergence diagnostics;
- solution interpolation;
- basic worked examples;
- automated tests.

The PINN backend and later capabilities are being developed on separate branches and are **not yet part of the stable `v0.1.0` release**.

---

## Requirements

- Python 3.10 or newer
- NumPy
- SciPy
- Matplotlib

Development and PINN branches additionally use:

- PyTorch
- pytest

---

## Stable installation

The current stable tagged release can be installed directly from GitHub:

```bash
python -m pip install \
"git+https://github.com/DrPrabir/pinns4bvp.git@v0.1.0"
```

Verify the installation:

```bash
python -c "import pinns4bvp; print('PINNs4BVP imported successfully')"
```

---

## Install from source

Clone the repository:

```bash
git clone https://github.com/DrPrabir/pinns4bvp.git
cd pinns4bvp
```

Create a virtual environment:

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install in editable mode:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

Editable installation is recommended for development.

---

## Quick start

Consider the BVP

\[
y''=-1,
\qquad
y(0)=0,
\qquad
y(1)=0.
\]

Its exact solution is

\[
y(x)=\frac{x(1-x)}{2}.
\]

Write the second-order equation as a first-order system:

\[
y_1'=y_2,
\qquad
y_2'=-1.
\]

Example:

```python
import numpy as np

from pinns4bvp import BVPProblem, solve


def equations(x, y, p):
    return np.vstack((
        y[1],
        -np.ones_like(x)
    ))


def boundary_conditions(ya, yb, p):
    return np.array([
        ya[0],
        yb[0]
    ])


problem = BVPProblem(
    equations=equations,
    boundary_conditions=boundary_conditions,
    domain=(0.0, 1.0),
    n_equations=2,
    variable_names=("y", "yp"),
    name="Linear test BVP",
)


solution = solve(
    problem,
    tol=1e-8
)


print(solution.summary())
```

Evaluate the solution on a custom grid:

```python
x = np.linspace(0.0, 1.0, 201)

y_num = solution.values("y", x)
y_exact = 0.5 * x * (1.0 - x)

max_error = np.max(np.abs(y_num - y_exact))

print("Maximum error:", max_error)
```

---

## Public API

The current stable workflow is centered around:

```python
from pinns4bvp import BVPProblem, solve
```

A typical problem definition is:

```python
problem = BVPProblem(
    equations=equations,
    boundary_conditions=bc,
    domain=(a, b),
    n_equations=n,
    variable_names=(...)
)
```

and is solved with:

```python
solution = solve(problem)
```

The returned `BVPSolution` provides access to convergence information and interpolated solution values.

---

## Examples

The repository includes examples for:

- a simple linear BVP;
- the nonlinear Bratu problem;
- the Blasius BVP.

These examples are intended as verification problems and usage demonstrations rather than application-specific restrictions on the library.

---

## Development roadmap

| Version | Goal |
|---|---|
| **v0.1** | Basic general BVP framework using SciPy backend |
| **v0.2** | Initial PyTorch PINN backend |
| **v0.3** | Robust PINN training: Adam + L-BFGS, diagnostics, reproducibility |
| **v0.4** | General benchmarking: numerical vs PINN vs exact |
| **v0.5** | Unknown parameters and eigenvalue BVPs |
| **v0.6** | Better mesh, initial guesses, residual diagnostics |
| **v0.7** | Parameter continuation and nonlinear solution families |
| **v0.8** | Higher-order convenience API, Jacobians, singular/problem utilities |
| **v0.9** | API stabilization, documentation, cross-platform testing |
| **v1.0** | Stable general-purpose BVP solver framework |

---

## Development branches

Development versions can be installed directly from their GitHub branches.

For example:

```bash
python -m pip install \
"git+https://github.com/DrPrabir/pinns4bvp.git@v0.5-unknown-parameters"
```

Development branches may contain experimental APIs and should not be assumed to be backward compatible before v1.0.

---

## Testing

For a source checkout, install the test dependencies and run:

```bash
python -m pip install pytest
pytest -v
```

Development branches may define additional optional dependencies.

---

## Documentation

Project documentation currently includes:

- `README.md` — project overview and quick start;
- `CHANGELOG.md` — version history;
- `DEVELOPMENT_NOTES.md` — implementation and development notes on newer branches;
- `THIRD_PARTY.md` — third-party software acknowledgements;
- `LICENSE` — MIT License.

A more complete user manual and peer-testing guide is being developed as the project approaches wider testing.

---

## Versioning

PINNs4BVP follows semantic-style versioning during development.

Stable tagged releases use versions such as:

```text
v0.1.0
v0.2.0
...
v1.0.0
```

Development snapshots may use versions such as:

```text
0.5.0.dev0
```

For reproducible research, users are encouraged to record the exact PINNs4BVP version or Git commit used.

---

## License

PINNs4BVP is released under the **MIT License**.

See:

```text
LICENSE
```

for the full license text.

Third-party dependencies retain their respective licenses. See:

```text
THIRD_PARTY.md
```

for acknowledgements.

---

## Disclaimer

PINNs4BVP is research software under active development.

Before v1.0:

- APIs may change;
- numerical behavior may be refined;
- PINN training behavior may depend on optimization settings;
- results should be independently verified for research-critical calculations.

Users are encouraged to compare results against analytical solutions, established numerical methods, or trusted benchmark data whenever possible.

---

## Contributing and feedback

Bug reports, feature suggestions, numerical test cases, and documentation feedback are welcome through the GitHub repository:

```text
https://github.com/DrPrabir/pinns4bvp
```

When reporting a numerical issue, please include:

- PINNs4BVP version or branch;
- Python version;
- operating system;
- complete problem definition;
- initial guess and mesh;
- solver settings;
- full error message or convergence output.

---

## Citation

A formal citation file will be added before the stable v1.0 release.

For now, when referring to development versions in research notes or testing reports, please include the repository URL and exact version or branch used.

---

## Acknowledgements

PINNs4BVP currently builds on established open-source scientific Python software, including NumPy, SciPy, Matplotlib, PyTorch on PINN-enabled development branches, and pytest for testing.

See `THIRD_PARTY.md` for details.

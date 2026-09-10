# PINNs4BVP v0.6 development snapshot

PINNs4BVP is a general-purpose Python framework for two-point boundary-value problems with classical collocation and physics-informed neural-network (PINN) backends.

**v0.6 focus:** better initial meshes, richer initial guesses, independent residual diagnostics, and explicit PINN device selection for CPU, CUDA, and Apple MPS.

> Status: alpha development snapshot. This is not yet a stable release.

## Installation for development

```bash
python -m pip install -e ".[dev]"
```

Classical-only use:

```bash
python -m pip install -e .
```

## What is new in v0.6

- `MeshConfig` with uniform, left-clustered, right-clustered, and Chebyshev/cosine-clustered meshes.
- Mesh spacing diagnostics through `mesh_quality(...)`.
- Initial guesses from arrays, callables, variable mappings, previous solutions, or interpolated source data.
- Backend-independent residual diagnostics through `sol.residual_report()`.
- Residual plotting through `sol.plot_residuals()`.
- PINN device choices: `cpu`, `cuda`, `mps`, and `auto`.
- Device availability helpers and explicit device metadata in PINN solutions.
- All v0.5 unknown-parameter and eigenvalue capabilities are retained.

## Mesh control

```python
from pinns4bvp.mesh import MeshConfig

mesh = MeshConfig(
    n_nodes=80,
    kind="chebyshev",
)

sol = solve(problem, mesh=mesh)
```

Supported mesh kinds are:

```text
uniform
left          # clusters near the left endpoint
quadratic     # backward-compatible alias of left with power=2
right         # clusters near the right endpoint
chebyshev     # clusters near both endpoints
```

For power-law clustering:

```python
mesh = MeshConfig(n_nodes=80, kind="left", power=3.0)
```

Inspect the initial mesh used by a classical solve:

```python
print(sol.metadata["initial_mesh_quality"].summary())
```

## Better initial guesses

The classical backend accepts several guess forms.

Zeros:

```python
sol = solve(problem, guess="zeros")
```

Callable:

```python
def guess(x):
    return np.vstack((x * (1 - x), 1 - 2*x))

sol = solve(problem, guess=guess)
```

Variable mapping:

```python
sol = solve(
    problem,
    guess={
        "y": lambda x: x * (1 - x),
        "yp": 0.0,
    },
)
```

Reuse a previous solution on a different mesh:

```python
from pinns4bvp.guess import guess_from_solution

sol1 = solve(problem, n_mesh=30)
sol2 = solve(
    problem,
    n_mesh=100,
    guess=guess_from_solution(sol1),
)
```

A previous `BVPSolution` may also be supplied directly as `guess=sol1`.

## Independent residual diagnostics

After either a classical or PINN solve:

```python
report = sol.residual_report(n_points=301)
print(report.summary())
```

The diagnostic residual is evaluated independently as

$$
R(x) = y'(x) - f(x,y,p).
$$

The report provides:

- global ODE RMS residual;
- global maximum absolute ODE residual;
- per-equation RMS and maximum residuals;
- maximum boundary-condition residual.

A residual plot is available with:

```python
sol.plot_residuals()
```

## PINN device selection

PINNs4BVP v0.6 supports the following device requests:

```python
from pinns4bvp.pinn import PINNConfig

cpu = PINNConfig(device="cpu", dtype="float64")
cuda = PINNConfig(device="cuda", dtype="float64")
mps = PINNConfig(device="mps", dtype="float32")
auto = PINNConfig(device="auto", dtype="float64")
```

### CPU

```python
PINNConfig(device="cpu", dtype="float64")
```

CPU remains the conservative default for small BVPs and reproducible float64 experiments.

### CUDA

```python
PINNConfig(device="cuda", dtype="float64")
```

CUDA is used only when the current PyTorch installation reports it as available. An unavailable explicit CUDA request raises a clear error.

### Apple MPS

```python
PINNConfig(device="mps", dtype="float32")
```

MPS is intended for supported Apple Silicon/macOS PyTorch installations. v0.6 deliberately requires float32 for an explicit MPS request rather than assuming float64 MPS support across supported environments.

### Automatic selection

```python
PINNConfig(device="auto", dtype="float32")
```

`auto` uses the following policy:

1. CUDA, when available;
2. MPS, when available and `dtype="float32"`;
3. CPU otherwise.

For `dtype="float64"`, `auto` prefers CUDA when available and otherwise uses CPU, preserving the requested precision instead of silently changing dtype.

Inspect device availability:

```python
from pinns4bvp.pinn import available_devices, device_summary

print(available_devices())
print(device_summary())
```

A PINN result records both the requested and resolved device:

```python
print(sol.metadata["device_requested"])
print(sol.metadata["device_resolved"])
```

## Unknown parameters and eigenvalue BVPs

v0.5 functionality remains available:

```python
from pinns4bvp import UnknownParameter

problem = BVPProblem(
    ...,
    unknown_parameters={"k": UnknownParameter(initial=3.0)},
)
```

Unknown parameters can be solved by both the SciPy collocation and PINN backends.

## Examples

Run the v0.6 diagnostics example:

```bash
python -m pinns4bvp.examples.residual_diagnostics
```

Inspect available PINN devices:

```bash
python -m pinns4bvp.examples.device_selection
```

Earlier examples remain available, including linear, Bratu, Blasius, benchmarking, and eigenvalue problems.

## Testing

```bash
pytest -v
```

The v0.6 development suite includes regression tests for all earlier capabilities plus mesh generation, interpolated/reused guesses, residual diagnostics, and device selection.

## Current limitations

- Classical callbacks use NumPy while PINN callbacks are PyTorch-native in the current alpha API.
- Unknown parameters are scalar and unconstrained.
- `mps` is intentionally restricted to `float32` in v0.6.
- Availability of CUDA/MPS depends on the installed PyTorch build and host hardware.
- Device-specific numerical trajectories can differ even with identical seeds.
- This release does not yet implement adaptive continuation or higher-order equation syntax.

## License

MIT. See `LICENSE` and `THIRD_PARTY.md`.

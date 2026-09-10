# PINNs4BVP v0.7 development snapshot

PINNs4BVP is a general-purpose Python framework for two-point boundary-value problems with classical collocation and physics-informed neural-network (PINN) backends.

**v0.7 focus:** natural parameter continuation, reusable nonlinear solution families, adaptive recovery from failed continuation steps, and experimental PINN warm starts.

> Status: alpha development snapshot. This is not yet a stable release.

## Installation for development

```bash
python -m pip install -e ".[dev]"
```

Classical-only use:

```bash
python -m pip install -e .
```

## What is new in v0.7

- `continue_parameter(...)` for one-parameter natural continuation.
- `BVPProblem.continue_parameter(...)` convenience method.
- `ContinuationConfig` for adaptive recovery, step limits, failure handling, and diagnostics.
- `SolutionFamily` and `ContinuationPoint` containers.
- Automatic reuse of the previous classical solution as the next initial guess.
- Optional proactive step subdivision through `max_step`.
- Adaptive step reduction after failed solves.
- Continuation can coexist with v0.5 unknown parameters/eigenvalue BVPs.
- PINN continuation can warm-start compatible network weights from the previous accepted PINN solution.
- Unknown PINN/BVP parameter estimates are also warm-started during continuation.
- Family-level tracking, profile plotting, summaries, and solve-time diagnostics.
- All v0.6 mesh, guess, residual-diagnostic, CPU/CUDA/MPS, and automatic-device features are retained.

## Basic continuation

Suppose a fixed parameter `lambda` is already declared in the problem:

```python
problem = BVPProblem(
    equations=equations,
    boundary_conditions=bc,
    domain=(0.0, 1.0),
    n_equations=2,
    parameters={"lambda": 0.0},
    variable_names=("y", "yp"),
)
```

Continue through a monotonic sequence:

```python
from pinns4bvp import continue_parameter

family = continue_parameter(
    problem,
    parameter="lambda",
    values=[0.0, 0.5, 1.0, 1.5, 2.0],
    solve_kwargs={"tol": 1e-8},
)

print(family.summary())
```

The original `problem` is not mutated. A parameter-specific problem copy is constructed for each solve.

The equivalent convenience form is:

```python
family = problem.continue_parameter(
    "lambda",
    [0.0, 0.5, 1.0, 1.5, 2.0],
    solve_kwargs={"tol": 1e-8},
)
```

## Reuse of previous solutions

For the classical backend, each accepted solution becomes the initial state guess for the next continuation point automatically:

```text
lambda_0 -> solve -> solution_0
                    |
                    +--> initial guess for lambda_1
                              |
                              +--> initial guess for lambda_2
```

If unknown BVP parameters are present, their solved values can also initialize the next point.

## Adaptive recovery

```python
from pinns4bvp import ContinuationConfig

config = ContinuationConfig(
    adaptive=True,
    min_step=1e-3,
    max_step=0.5,
    reduction_factor=0.5,
    max_retries=8,
    stop_on_failure=False,
)

family = continue_parameter(
    problem,
    "lambda",
    values,
    config=config,
)
```

If a requested step fails, v0.7 can insert smaller intermediate points and retry the target from the closest accepted state.

`max_step` can also be used to proactively subdivide a large requested jump.

## SolutionFamily

A continuation returns a structured family rather than a bare list:

```python
family.success
family.values
family.solutions
family.requested_results
family.failed_points
family.diagnostics
```

Retrieve a solution at a particular parameter value:

```python
sol = family.solution_at(1.5)
```

Track one state quantity at a fixed position:

```python
parameter_values, response = family.track("y", x=0.5)
```

Apply an arbitrary scalar function to accepted solutions:

```python
parameter_values, response = family.evaluate(
    lambda sol: sol.values("y", 0.5)
)
```

Plot profiles:

```python
family.plot_profiles("y")
```

Plot a tracked quantity:

```python
family.plot_track("y", x=0.5)
```

## Continuation diagnostics

```python
print(family.diagnostics.summary())
```

The report includes:

- number of requested values;
- requested values converged/failed;
- total solve attempts;
- intermediate recovery attempts;
- intermediate accepted points;
- total solve time.

When enabled, each continuation point may also hold an independent v0.6 residual report:

```python
point.residual_report
```

## Fixed continuation parameters versus unknown parameters

These are distinct concepts.

A fixed continuation parameter is deliberately varied:

```python
parameters={"mu": 1.0}
```

An unknown parameter is solved as part of the BVP:

```python
unknown_parameters={"k": UnknownParameter(3.0)}
```

They may coexist. For example, v0.7 can vary `mu` while solving an eigen-parameter `k` at every continuation point.

```python
family = continue_parameter(
    problem,
    "mu",
    [1.0, 1.25, 1.5, 2.0],
)

for point in family.requested_results:
    if point.accepted:
        print(point.value, point.solution.parameters["k"])
```

## PINN continuation and warm starts

For a PINN-enabled problem:

```python
family = continue_parameter(
    problem,
    "lambda",
    [0.0, 0.5, 1.0],
    method="pinn",
    pinn_config=config,
)
```

By default v0.7 reuses compatible trained network weights from the previous accepted PINN solution. The solution metadata records this:

```python
sol.metadata["warm_start_used"]
```

PINN warm starting can be disabled:

```python
ContinuationConfig(pinn_warm_start=False)
```

Because PINN `success` uses a strict residual-based criterion in the current alpha API, advanced users may provide a custom continuation acceptance rule:

```python
family = continue_parameter(
    problem,
    "lambda",
    values,
    method="pinn",
    pinn_config=config,
    accept_solution=lambda sol: sol.residual_report().ode_rms < 1e-3,
)
```

Use such overrides carefully and record the criterion in reproducible studies.

## Device selection

v0.6 device support remains available for PINN continuation:

```python
from pinns4bvp.pinn import PINNConfig

cpu = PINNConfig(device="cpu", dtype="float64")
cuda = PINNConfig(device="cuda", dtype="float64")
mps = PINNConfig(device="mps", dtype="float32")
auto = PINNConfig(device="auto", dtype="float32")
```

## Examples

Classical linear continuation:

```bash
python -m pinns4bvp.examples.continuation_linear_parameter
```

Nonlinear Bratu continuation:

```bash
python -m pinns4bvp.examples.continuation_bratu
```

Continuation with a simultaneously solved eigen-parameter:

```bash
python -m pinns4bvp.examples.continuation_eigenvalue
```

Experimental PINN warm-start continuation:

```bash
python -m pinns4bvp.examples.pinn_continuation_linear
```

Earlier examples for benchmarking, residual diagnostics, device selection, eigenvalue BVPs, Bratu, Blasius, and linear problems remain available.

## Testing

```bash
python -m pytest -v
```

The v0.7 suite retains all earlier regression tests and adds tests for:

- basic continuation;
- family tracking;
- original-problem immutability;
- continuation with unknown parameters;
- adaptive step recovery;
- PINN model warm-start mechanics.

## Current limitations

- v0.7 implements **natural continuation in one fixed scalar parameter**.
- Requested continuation values must be strictly monotonic.
- Full pseudo-arclength continuation is not implemented.
- Turning-point detection and automatic bifurcation classification are not implemented.
- PINN warm starting is experimental and requires a compatible network architecture.
- Classical callbacks remain NumPy-based while PINN callbacks are PyTorch-native in the current alpha API.
- Unknown parameters remain scalar and unconstrained.
- CUDA/MPS availability depends on the installed PyTorch build and host hardware.

## License

MIT. See `LICENSE` and `THIRD_PARTY.md`.

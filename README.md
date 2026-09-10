# PINNs4BVP v0.8 development snapshot

PINNs4BVP is a general-purpose Python framework for two-point boundary-value problems (BVPs) with classical collocation and physics-informed neural-network (PINN) backends.

**v0.8 focus:** a higher-order mathematical formulation API, automatic first-order compilation for classical and PINN solvers, simultaneous/mixed-order systems, continuation result-retention controls, and pre-v1.0 API refinement.

> Status: alpha development snapshot. This is not yet a stable release.

## Installation for development

```bash
python -m pip install -e ".[dev]"
```

Classical-only use:

```bash
python -m pip install -e .
```

## What is new in v0.8

- Default mathematical symbols `y(x)` for ordinary single-equation BVPs.
- `d(y)`, `d2(y)`, `d3(y)`, `d4(y)`, and general `derivative(y, order=n)` notation.
- `Parameter`, `Equation`, and `HigherOrderBVP` high-level formulation objects.
- Automatic conversion of higher-order equations to the established first-order `BVPProblem` representation.
- One high-level equation definition automatically generates both NumPy and PyTorch callbacks.
- Explicit `IndependentVariable` and `DependentVariable` declarations for custom symbols and simultaneous systems.
- Coupled nonlinear systems and mixed differential orders.
- Linear coupled highest-derivative systems are solved automatically during compilation/evaluation.
- Mathematical endpoint conditions such as `y.at(0) == 0` and `d(y).at(1) == 1`.
- High-level unknown/eigen parameters through `Parameter(..., initial=..., unknown=True)`.
- Automatic boundary-condition count validation.
- Continuation with `start=`, `stop=`, and `step=` in addition to explicit value lists.
- Continuation storage policies: `save="all"`, `save="requested"`, and `save="final"`.
- `BVPProblem.continue_to(...)` for continuation used only as a route to a final target solution.
- `BVPSolution.derivative(...)` convenience method.
- Optional analytical Jacobian hooks and a SciPy-compatible singular matrix on the low-level `BVPProblem` API.
- All v0.1-v0.7 public workflows remain supported.

## Default `y(x)` notation

For a standard equation

$$
y'' + a y' + b y = c,
$$

the user does not need to declare `x` or `y` explicitly:

```python
from pinns4bvp import Equation, HigherOrderBVP, Parameter, d, d2, y


a = Parameter("a", value=2.0)
b = Parameter("b", value=3.0)
c = Parameter("c", value=1.0)

problem = HigherOrderBVP(
    equation=Equation(
        d2(y) + a*d(y) + b*y,
        c,
    ),
    boundary_conditions=[
        y.at(0.0) == 0.0,
        y.at(1.0) == 1.0,
    ],
    domain=(0.0, 1.0),
)

solution = problem.solve(tol=1e-9)
```

In the default notation:

```text
y      -> dependent variable
d(y)   -> dy/dx
d2(y)  -> d2y/dx2
x      -> independent variable
```

`Equation(lhs, rhs)` is used for governing differential equations. The `==` relation is used for boundary conditions.

## Explicit variable names

Users can override the defaults when needed:

```python
from pinns4bvp import IndependentVariable, DependentVariable, Equation, HigherOrderBVP, d2

r = IndependentVariable("r")
T = DependentVariable("T", order=2)

problem = HigherOrderBVP(
    equation=Equation(d2(T, r), -2.0),
    boundary_conditions=[
        T.at(0.0) == 0.0,
        T.at(1.0) == 0.0,
    ],
    domain=(0.0, 1.0),
)
```

## Simultaneous coupled systems

Multiple dependent variables can be declared explicitly:

```python
from pinns4bvp import DependentVariable, Equation, HigherOrderBVP, d2

u = DependentVariable("u", order=2)
v = DependentVariable("v", order=2)

problem = HigherOrderBVP(
    equations=[
        Equation(d2(u) - 2*u*v, 0.0),
        Equation(d2(v) - 6*u**2*v, 0.0),
    ],
    boundary_conditions=[
        u.at(0.0) == 1.0,
        u.at(1.0) == 0.5,
        v.at(0.0) == 1.0,
        v.at(1.0) == 0.25,
    ],
    domain=(0.0, 1.0),
)
```

The internal first-order state is generated automatically:

```text
u, du/dx, v, dv/dx
```

## Mixed-order systems

Different equations may have different differential orders:

```python
u = DependentVariable("u", order=3)
v = DependentVariable("v", order=2)

problem = HigherOrderBVP(
    equations=[
        Equation(d3(u) + d2(v), 8.0),
        Equation(d3(u) - d2(v), 4.0),
    ],
    boundary_conditions=[
        u.at(0.0) == 0.0,
        d(u).at(0.0) == 0.0,
        u.at(1.0) == 1.0,
        v.at(0.0) == 0.0,
        v.at(1.0) == 1.0,
    ],
    domain=(0.0, 1.0),
)
```

The resulting internal state is:

```text
u, du/dx, d2u/dx2, v, dv/dx
```

The v0.8 compiler can also handle a linear system in the highest derivatives, as in the example above.

## Parameters

### Known parameter

```python
a = Parameter("a", value=2.0)
```

### Unknown/eigen parameter

```python
k = Parameter(
    "k",
    initial=3.0,
    unknown=True,
)
```

For example,

$$
y'' + k^2y = 0
$$

can be written directly as:

```python
k = Parameter("k", initial=3.0, unknown=True)

problem = HigherOrderBVP(
    equation=Equation(d2(y) + k**2*y, 0.0),
    boundary_conditions=[
        y.at(0.0) == 0.0,
        y.at(1.0) == 0.0,
        d(y).at(0.0) == 1.0,
    ],
    domain=(0.0, 1.0),
)
```

This compiles to the v0.5 unknown-parameter machinery internally. The low-level `UnknownParameter` dictionary API remains supported.

## Boundary-condition counting

If the compiled first-order system contains `n` states and `k` unknown scalar parameters, v0.8 requires exactly

$$
N_{BC} = n + k
$$

independent boundary/normalization conditions.

Incorrect counts are rejected before the numerical solve begins.

## One problem definition for classical and PINN backends

A high-level `HigherOrderBVP` automatically creates both backend representations. The user no longer needs separate NumPy and PyTorch equation callbacks for high-level problems.

Classical:

```python
solution = problem.solve(
    method="collocation",
    tol=1e-9,
)
```

PINN:

```python
from pinns4bvp.pinn import PINNConfig

config = PINNConfig(
    hidden_layers=(32, 32),
    n_collocation=80,
    adam_epochs=2000,
    use_lbfgs=True,
    device="auto",
    dtype="float32",
)

solution = problem.solve(
    method="pinn",
    pinn_config=config,
)
```

The low-level `BVPProblem` API still permits separate NumPy/PyTorch callbacks for advanced or unsupported formulations.

## Mathematical functions

The high-level expression API currently provides backend-neutral functions:

```python
exp(...)
sin(...)
cos(...)
tanh(...)
sqrt(...)
log(...)
```

For example, the Bratu equation can be represented as:

```python
lam = Parameter("lambda", value=1.0)
Equation(d2(y) + lam*exp(y), 0.0)
```

## Solution access

Compiled state names remain available:

```python
solution.variable_names
```

For a second-order `y` problem this is typically:

```text
('y', 'd(y)/dx')
```

Evaluate the physical variable:

```python
solution.values("y", x_eval)
```

Evaluate derivatives directly:

```python
solution.derivative("y", order=1, x=x_eval)
```

## Parameter continuation in v0.8

The v0.7 explicit-value syntax is unchanged:

```python
family = problem.continue_parameter(
    "lambda",
    [0.0, 0.5, 1.0, 1.5],
)
```

A high-level `Parameter` object can also be passed directly:

```python
family = problem.continue_parameter(
    lam,
    start=0.0,
    stop=2.0,
    step=0.25,
)
```

### Continuation result storage

Three retention policies are available:

```python
save="all"
save="requested"
save="final"
```

- `"all"`: retain full solutions for requested and adaptive intermediate points. This is the default to preserve v0.7 behavior.
- `"requested"`: retain full solutions only at user-requested parameter values.
- `"final"`: retain only the final requested full solution.

All modes preserve lightweight status/residual history through:

```python
family.history
```

For large PINN studies, `save="requested"` or `save="final"` can reduce the size of the returned solution family.

## Continue only to a final target

When continuation is only a numerical strategy to reach one target value:

```python
solution = problem.continue_to(
    lam,
    target=10.0,
    start=0.0,
    step=0.25,
    method="collocation",
)
```

`continue_to()` uses `save="final"` internally and returns the final `BVPSolution` directly. Lightweight continuation diagnostics are attached to its metadata.

## PINN warm starts and devices

All v0.7 PINN continuation behavior is retained. Compatible trained network weights can warm-start the next continuation point.

Device selection from v0.6 remains available:

```python
PINNConfig(device="cpu", dtype="float64")
PINNConfig(device="cuda", dtype="float64")
PINNConfig(device="mps", dtype="float32")
PINNConfig(device="auto", dtype="float32")
```

## Optional Jacobian hooks

Advanced low-level `BVPProblem` users may supply analytical callbacks:

```python
problem = BVPProblem(
    equations=fun,
    boundary_conditions=bc,
    domain=(0.0, 1.0),
    n_equations=2,
    equations_jacobian=fun_jac,
    boundary_jacobian=bc_jac,
)
```

These callbacks are forwarded to SciPy's `solve_bvp` backend. Automatic symbolic Jacobian generation for arbitrary high-level expressions is **not** claimed in v0.8.

## Singular-term hook

The low-level API also accepts a SciPy-compatible singular matrix:

```python
problem = BVPProblem(
    ...,
    singular_matrix=S,
)
```

`S` must have shape `(n_equations, n_equations)`. Users should follow the mathematical requirements of SciPy's singular-term formulation.

## Backward compatibility

v0.8 is intentionally additive. Existing first-order code remains valid:

```python
problem = BVPProblem(
    equations=equations,
    boundary_conditions=bc,
    domain=(0.0, 1.0),
    n_equations=2,
    variable_names=("y", "yp"),
)

solution = solve(problem)
```

Existing v0.2-v0.7 PINN callbacks, unknown parameters, mesh configuration, residual diagnostics, benchmarking, continuation, and PINN warm-start APIs remain available.

The v0.8 test suite runs all inherited regression tests together with the new formulation tests.

## Examples

```bash
python -m pinns4bvp.examples.higher_order_linear_bvp
python -m pinns4bvp.examples.higher_order_nonlinear_bvp
python -m pinns4bvp.examples.coupled_system_bvp
python -m pinns4bvp.examples.mixed_order_system
python -m pinns4bvp.examples.higher_order_eigenvalue
python -m pinns4bvp.examples.higher_order_continuation
python -m pinns4bvp.examples.pinn_higher_order_bvp
```

Earlier examples remain included.

## Testing

```bash
python -m pytest -v
```

The v0.8 regression suite covers:

- all inherited v0.1-v0.7 tests;
- default `y(x)` higher-order formulation;
- explicit independent/dependent variables;
- nonlinear coupled systems;
- mixed-order systems;
- coupled leading derivatives;
- high-level eigenvalue problems;
- automatic PyTorch callback generation;
- actual high-level PINN execution;
- continuation storage policies;
- `continue_to()`;
- low-level Jacobian forwarding;
- singular-matrix validation;
- formulation error checking.

## Current limitations

- The high-level formulation targets explicit ODE BVPs that can be reduced to a first-order system.
- The highest derivatives must enter the governing equations linearly. A linear coupled system of highest derivatives is supported; nonlinear dependence on a highest derivative is rejected.
- Boundary conditions in the high-level API are restricted to the two domain endpoints.
- Boundary conditions on the highest derivative itself are not currently supported by the high-level compiler.
- Differential-algebraic equation systems are outside the v0.8 scope.
- Automatic symbolic Jacobian generation is not implemented.
- Natural one-parameter continuation remains the supported continuation model; pseudo-arclength/bifurcation tools remain future work.
- Low-level callbacks remain available for formulations not expressible through the high-level v0.8 API.

## License

MIT. See `LICENSE` and `THIRD_PARTY.md`.

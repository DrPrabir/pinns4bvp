# v0.8 Higher-Order API Reference

This document summarizes the new formulation layer. The low-level `BVPProblem` interface remains valid and is the internal representation used by `HigherOrderBVP`.

## Default scalar problem

```python
from pinns4bvp import Equation, HigherOrderBVP, Parameter, d, d2, y

a = Parameter("a", value=2.0)
b = Parameter("b", value=3.0)
c = Parameter("c", value=1.0)

problem = HigherOrderBVP(
    equation=Equation(d2(y) + a*d(y) + b*y, c),
    boundary_conditions=[y.at(0) == 0, y.at(1) == 1],
    domain=(0, 1),
)
```

The default independent variable is `x`; the default dependent variable is `y`.

## Derivatives

```python
d(y)                    # first derivative
d2(y)                   # second derivative
d3(y)                   # third derivative
d4(y)                   # fourth derivative
derivative(y, order=7)  # arbitrary positive order
```

With an explicit independent variable:

```python
r = IndependentVariable("r")
T = DependentVariable("T", order=2)
d2(T, r)
```

## Known and unknown parameters

```python
alpha = Parameter("alpha", value=1.5)
k = Parameter("k", initial=3.0, unknown=True)
```

High-level unknown parameters compile to the established low-level `UnknownParameter` machinery.

## Simultaneous systems

```python
u = DependentVariable("u", order=2)
v = DependentVariable("v", order=2)

problem = HigherOrderBVP(
    equations=[
        Equation(d2(u) + u*v, 0),
        Equation(d2(v) - u**2, 0),
    ],
    boundary_conditions=[
        u.at(0) == 0,
        u.at(1) == 1,
        v.at(0) == 1,
        v.at(1) == 0,
    ],
    domain=(0, 1),
)
```

## Supported expression operations

Arithmetic:

```text
+  -  *  /  **  unary -
```

Functions:

```text
exp, sin, cos, tanh, sqrt, log
```

The highest derivatives must occur linearly, though multiple highest derivatives may be coupled through a linear coefficient matrix.

## Boundary conditions

Use `.at(endpoint)` for state and derivative values:

```python
y.at(0) == 0
d(y).at(1) == 1
u.at(0) == v.at(1)
```

Only the two domain endpoints are accepted in v0.8.

## Backend behavior

The compiled `BVPProblem` contains both NumPy and PyTorch callbacks. Therefore the same high-level problem can be passed to:

```python
problem.solve(method="collocation")
problem.solve(method="pinn", pinn_config=config)
```

## Continuation

Existing explicit values:

```python
family = problem.continue_parameter("lambda", [0, 0.5, 1.0])
```

Generated range:

```python
family = problem.continue_parameter(
    lam,
    start=0,
    stop=2,
    step=0.25,
    save="requested",
)
```

Retention options:

```text
all        retain every full solution
requested  retain user-requested full solutions
final      retain only the final requested full solution
```

The compatibility default is `all`.

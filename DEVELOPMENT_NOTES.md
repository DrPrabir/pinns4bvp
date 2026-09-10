# PINNs4BVP v0.8 development notes

## Scope

v0.8 adds a high-level higher-order formulation layer while deliberately retaining the established first-order `BVPProblem` as the internal solver contract. The design goal is one mathematical problem definition that can feed both classical NumPy/SciPy and PyTorch/PINN backends.

## Architecture

The new high-level flow is:

```text
Higher-order expression API
        |
        v
expression validation / state layout
        |
        v
automatic first-order compiler
        |
        v
BVPProblem
   |         |
 SciPy      PINN
```

`HigherOrderBVP(...)` is intentionally a constructor/factory that returns the established `BVPProblem`. This avoids a parallel solver hierarchy and preserves compatibility with diagnostics, benchmarking, mesh/guess handling, unknown parameters, and continuation.

## Default notation

The module exports default `x` and `y` symbols. `d(y)` therefore means `dy/dx` and `d2(y)` means `d2y/dx2`. Explicit `IndependentVariable` and `DependentVariable` objects are available when users need different symbols or simultaneous systems.

## Differential-order inference

For a dependent variable declared with `order=None`, the compiler infers the order from the highest derivative appearing in the formulation. Explicit orders are recommended for simultaneous/mixed-order systems and are validated against used derivatives.

## First-order conversion

For each variable of order `m`, the compiler creates states corresponding to derivative orders `0` through `m-1`. The governing equations determine the order-`m` derivatives.

The compiler supports equations affine in the highest derivatives. For simultaneous systems, it forms a coefficient matrix for the highest derivatives and solves the resulting small linear system pointwise. Nonlinear dependence on a highest derivative is rejected with a formulation error.

This supports systems such as:

```text
u''' + v'' = f1(...)
u''' - v'' = f2(...)
```

without manual isolation by the user.

## Backend-neutral expressions

Arithmetic expression trees are evaluated against either NumPy arrays or PyTorch tensors. High-level problems therefore generate both `equations`/`boundary_conditions` and `pinn_equations`/`pinn_boundary_conditions` automatically.

The current backend-neutral function set is `exp`, `sin`, `cos`, `tanh`, `sqrt`, and `log`.

## Boundary conditions

High-level boundary conditions use explicit endpoint location nodes created by `.at(...)`. v0.8 validates that all locations are the two domain endpoints and that the number of residuals equals:

```text
number of compiled first-order states + number of unknown scalar parameters
```

Boundary conditions on derivative orders below the governing order are supported. Boundary conditions directly on the highest derivative remain outside the current high-level compiler.

## Parameters

High-level `Parameter("a", value=...)` objects compile to fixed entries in `BVPProblem.parameters`. `Parameter("k", initial=..., unknown=True)` compiles to the existing v0.5 `UnknownParameter` representation.

The original low-level `UnknownParameter` API is unchanged.

## Continuation retention

v0.8 adds three returned-family storage policies:

- `all`: retain requested and adaptive intermediate full solutions; this is the compatibility default.
- `requested`: discard full adaptive-intermediate solutions after continuation completes.
- `final`: retain only the final requested full solution.

All attempted points keep lightweight status, timing, warm-start, and residual metrics through `SolutionFamily.history`.

`continue_to(...)` uses `save="final"` and returns the final `BVPSolution` directly.

## Jacobians and singular systems

The low-level `BVPProblem` now accepts optional `equations_jacobian` and `boundary_jacobian` callbacks, which are adapted and forwarded to SciPy's collocation backend. A SciPy-compatible `singular_matrix` may also be provided.

v0.8 does not claim automatic symbolic Jacobian generation for arbitrary high-level expressions. SciPy's own numerical Jacobian estimation remains the default when analytical callbacks are absent.

## Backward compatibility policy

All public regression tests inherited from v0.1-v0.7 are run unchanged. New features are additive. Existing first-order callbacks, separate PINN callbacks, unknown parameters, mesh and guess APIs, residual diagnostics, benchmarking, device selection, and continuation remain supported.

The next release, v0.9, should focus on API freeze, documentation consistency, packaging cleanup, CI, cross-platform tests, and release-candidate hardening rather than major new mathematics.

## Deliberate v0.8 limitations

v0.8 does not implement:

- differential-algebraic equations;
- nonlinear equations in the highest derivatives;
- interior/multipoint boundary conditions in the high-level API;
- automatic symbolic Jacobian generation;
- pseudo-arclength continuation;
- automatic fold/bifurcation detection;
- stability analysis or branch switching.

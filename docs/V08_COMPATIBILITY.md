# v0.8 Backward Compatibility

PINNs4BVP v0.8 adds a higher-level mathematical front end but does not replace the original first-order API.

## Still supported

- `BVPProblem(...)` first-order NumPy callbacks.
- Separate `pinn_equations` and `pinn_boundary_conditions` callbacks.
- `solve(problem, method="collocation")` and `solve(problem, method="pinn")`.
- v0.5 `UnknownParameter` dictionaries.
- v0.6 `MeshConfig`, reusable solution guesses, residual reports, and device selection.
- v0.7 `continue_parameter`, `ContinuationConfig`, `SolutionFamily`, adaptive continuation, and PINN warm starts.

## Additive v0.8 conveniences

- `problem.solve(...)` delegates to the public `solve(...)` function.
- `HigherOrderBVP(...)` compiles to an ordinary `BVPProblem`.
- `continue_parameter` also accepts `start/stop/step` and named high-level `Parameter` objects.
- `save="all"` remains the default so full adaptive intermediate solutions continue to be retained unless users explicitly request a smaller result set.

## Regression expectation

All inherited v0.1-v0.7 tests should pass unchanged before a v0.8 snapshot is distributed for peer testing.

# Changelog

## 0.1.0 - 2026-09-09

Initial alpha release.

### Added
- `BVPProblem` for defining nonlinear two-point boundary-value problems.
- `solve()` public solver interface.
- `BVPSolution` wrapper independent of the SciPy result object.
- SciPy `solve_bvp` backend.
- Initial mesh utilities.
- Initial-guess utilities.
- Convergence diagnostics.
- Linear BVP, Bratu, and Blasius examples.
- Basic automated tests.

### Planned
- Parameter continuation.
- Semi-infinite-domain/far-field assistance.
- Research-oriented wall quantities and tabulation.
- Multiple-solution/branch utilities.
- PINN backend and benchmarking.

# PINNs4BVP v0.7 development notes

## Scope

v0.7 adds natural one-parameter continuation and solution-family management while retaining the general-purpose BVP design.

The continuation parameter must be a fixed scalar in `problem.parameters`. Unknown parameters introduced in v0.5 are solved at every continuation point and may be initialized from the preceding accepted solution.

## Core continuation objects

- `ContinuationConfig`: stepping, retry, residual, and warm-start policy.
- `ContinuationPoint`: one attempted parameter value and its solver outcome.
- `SolutionFamily`: requested values, all attempts, accepted solutions, tracking, plotting, and diagnostics.
- `continue_parameter(...)`: orchestration function.

## Classical continuation

For an accepted classical solution, the complete state is reused through the v0.6 previous-solution guess mechanism. This is backend-independent at the `BVPSolution` level and avoids exposing SciPy internals.

## Adaptive recovery

v0.7 uses natural continuation rather than pseudo-arclength continuation. When a target fails and an accepted source solution exists, the interval is reduced according to `reduction_factor` until the target can be approached or retry/minimum-step limits are reached.

An optional `max_step` proactively inserts intermediate points before a large jump.

All attempts are recorded; `SolutionFamily.requested_results` reports the final outcome for each user-requested value.

## Unknown parameters during continuation

The fixed continuation parameter and unknown BVP parameters are deliberately distinct. At each new point, unknown-parameter initial values may be replaced by the values recovered from the preceding accepted solution. The original `BVPProblem` is never mutated.

## PINN warm starts

v0.7 extends the PINN backend with an optional `pinn_warm_start` solution. Compatible network `state_dict` values are copied into a newly constructed model after device/dtype resolution. Solved unknown parameters are also reused as initial trainable values.

This keeps each continuation point as an independent `BVPSolution` while allowing efficient neural continuation.

Warm starts require the same domain, number of equations, and compatible network architecture.

## Acceptance policy

By default a continuation point is accepted only when `solution.success` is true. This is intentionally conservative. A custom `accept_solution(solution)` callback is available for experimental PINN studies with explicitly documented residual criteria.

## Deliberate v0.7 limitations

v0.7 does not implement:

- pseudo-arclength continuation;
- automatic fold/turning-point traversal;
- bifurcation classification;
- stability analysis;
- two-parameter continuation surfaces;
- automatic branch switching.

These capabilities require a more specialized continuation formulation and are intentionally outside this release.

# PINNs4BVP v0.6 development notes

v0.6 is a reliability and usability release. It deliberately does not add a new mathematical problem class. Instead it improves the infrastructure needed before parameter continuation and advanced BVP formulations.

## Mesh design

The classical backend continues to rely on the underlying adaptive collocation solver after initialization. v0.6 improves the **initial** mesh only. `MeshConfig` provides reproducible mesh definitions and the solver records a `MeshQualityReport` in solution metadata.

The supported initial mappings are intentionally generic: uniform, one-sided power clustering, and cosine/Chebyshev-style endpoint clustering.

## Guess design

`create_initial_guess(...)` now accepts previous solutions and interpolated source data. This is intentionally useful for the upcoming continuation release: a converged solution can be transferred to a different initial mesh without application-specific logic.

No attempt is made to infer a universal "physical" automatic guess from boundary conditions, because such inference is not reliable for general nonlinear BVPs.

## Residual diagnostics

`ResidualReport` evaluates

$$
y'(x)-f(x,y,p)
$$

using the backend-independent solution evaluator. The diagnostic is separate from SciPy's internal residual estimates and from the PINN training loss. This distinction is useful for cross-backend verification.

The classical NumPy callback is used to evaluate the right-hand side even for a PINN solution, providing an independent post-training check.

## Device policy

The public PINN choices are `cpu`, `cuda`, `mps`, and `auto`.

- CPU always resolves when PyTorch is installed.
- CUDA requires `torch.cuda.is_available()`.
- MPS requires a PyTorch build with an available MPS backend.
- Explicit MPS uses float32 in v0.6; float64 MPS is deliberately not assumed.
- `auto + float32`: CUDA -> MPS -> CPU.
- `auto + float64`: CUDA -> CPU.

This policy avoids silently reducing requested precision.

Device availability is an execution-environment property; tests therefore verify the selection logic without requiring CUDA or MPS hardware.

## Deliberate limitations

- No automatic adaptive initial-mesh optimizer beyond the backend's existing adaptive collocation process.
- No data-driven inverse-problem API.
- No full branch continuation yet.
- No symbolic/higher-order equation parser.
- No automatic dtype conversion to make MPS requests succeed.

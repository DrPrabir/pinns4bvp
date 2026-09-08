from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pinns4bvp.problem import BVPProblem


@dataclass(frozen=True, slots=True)
class ConvergenceReport:
    success: bool
    status: int
    message: str
    n_iterations: int | None
    n_nodes: int
    max_rms_residual: float | None
    bc_residual_inf: float

    def summary(self) -> str:
        residual = (
            "n/a" if self.max_rms_residual is None else f"{self.max_rms_residual:.3e}"
        )
        return (
            "Convergence report\n"
            "------------------\n"
            f"Success          : {self.success}\n"
            f"Status           : {self.status}\n"
            f"Iterations       : {self.n_iterations}\n"
            f"Mesh nodes       : {self.n_nodes}\n"
            f"Max RMS residual : {residual}\n"
            f"BC residual (inf): {self.bc_residual_inf:.3e}\n"
            f"Message          : {self.message}"
        )


def build_convergence_report(problem: BVPProblem, scipy_result) -> ConvergenceReport:
    bc_residual = problem.evaluate_boundary_conditions(
        scipy_result.y[:, 0], scipy_result.y[:, -1]
    )
    rms = getattr(scipy_result, "rms_residuals", None)
    max_rms = None if rms is None or len(rms) == 0 else float(np.max(np.abs(rms)))

    return ConvergenceReport(
        success=bool(scipy_result.success),
        status=int(scipy_result.status),
        message=str(scipy_result.message),
        n_iterations=getattr(scipy_result, "niter", None),
        n_nodes=int(scipy_result.x.size),
        max_rms_residual=max_rms,
        bc_residual_inf=float(np.max(np.abs(bc_residual))),
    )

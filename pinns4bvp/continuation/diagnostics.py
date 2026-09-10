"""Diagnostics for parameter-continuation runs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ContinuationDiagnostics:
    parameter_name: str
    n_requested: int
    n_attempts: int
    n_requested_converged: int
    n_requested_failed: int
    n_intermediate_attempts: int
    n_intermediate_converged: int
    total_solve_time: float

    @property
    def success(self) -> bool:
        return self.n_requested_failed == 0

    def summary(self) -> str:
        return "\n".join(
            [
                "Continuation diagnostics",
                "------------------------",
                f"Parameter             : {self.parameter_name}",
                f"Requested values      : {self.n_requested}",
                f"Requested converged   : {self.n_requested_converged}",
                f"Requested failed      : {self.n_requested_failed}",
                f"Total solve attempts  : {self.n_attempts}",
                f"Intermediate attempts : {self.n_intermediate_attempts}",
                f"Intermediate converged: {self.n_intermediate_converged}",
                f"Total solve time (s)  : {self.total_solve_time:.6g}",
                f"Overall success       : {self.success}",
            ]
        )

"""Structured diagnostics for PINN optimization."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class PINNTrainingRecord:
    step: int
    stage: str
    total_loss: float
    ode_loss: float
    bc_loss: float
    grad_norm: float | None = None
    parameters: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class PINNTrainingHistory:
    """Loss, gradient, and unknown-parameter history during optimization."""

    records: list[PINNTrainingRecord] = field(default_factory=list)
    stop_reason: str | None = None
    adam_steps: int = 0
    lbfgs_evaluations: int = 0

    def append(
        self,
        *,
        step: int,
        stage: str,
        total_loss: float,
        ode_loss: float,
        bc_loss: float,
        grad_norm: float | None = None,
        parameters: dict[str, float] | None = None,
    ) -> None:
        self.records.append(
            PINNTrainingRecord(
                step=int(step),
                stage=str(stage),
                total_loss=float(total_loss),
                ode_loss=float(ode_loss),
                bc_loss=float(bc_loss),
                grad_norm=None if grad_norm is None else float(grad_norm),
                parameters=dict(parameters or {}),
            )
        )

    @property
    def final(self) -> PINNTrainingRecord | None:
        return self.records[-1] if self.records else None

    @property
    def total_loss(self) -> list[float]:
        return [record.total_loss for record in self.records]

    @property
    def ode_loss(self) -> list[float]:
        return [record.ode_loss for record in self.records]

    @property
    def bc_loss(self) -> list[float]:
        return [record.bc_loss for record in self.records]

    def parameter_history(self, name: str) -> list[float]:
        return [record.parameters[name] for record in self.records if name in record.parameters]

    def summary(self) -> str:
        if self.final is None:
            return "PINN training history: no records"
        lines = [
            "PINN training report",
            "--------------------",
            f"Adam steps       : {self.adam_steps}",
            f"L-BFGS evaluations: {self.lbfgs_evaluations}",
            f"Final total loss : {self.final.total_loss:.3e}",
            f"Final ODE loss   : {self.final.ode_loss:.3e}",
            f"Final BC loss    : {self.final.bc_loss:.3e}",
            f"Stop reason      : {self.stop_reason or 'completed'}",
        ]
        if self.final.parameters:
            lines.extend(["", "Final unknown parameters"])
            for name, value in self.final.parameters.items():
                lines.append(f"{name:<16}: {value:.12g}")
        return "\n".join(lines)

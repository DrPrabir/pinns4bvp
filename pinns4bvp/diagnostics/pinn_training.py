"""Training diagnostics for the forthcoming PINN backend."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class PINNTrainingHistory:
    """Loss history recorded during PINN optimization."""

    epoch: list[int] = field(default_factory=list)
    total_loss: list[float] = field(default_factory=list)
    ode_loss: list[float] = field(default_factory=list)
    bc_loss: list[float] = field(default_factory=list)

    def append(
        self,
        epoch: int,
        total_loss: float,
        ode_loss: float,
        bc_loss: float,
    ) -> None:
        self.epoch.append(int(epoch))
        self.total_loss.append(float(total_loss))
        self.ode_loss.append(float(ode_loss))
        self.bc_loss.append(float(bc_loss))

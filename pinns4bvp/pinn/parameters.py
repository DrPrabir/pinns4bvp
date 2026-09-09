"""Trainable unknown parameters for the PINN backend."""

from __future__ import annotations


class PINNUnknownParameterSet:
    """Small container of scalar ``torch.nn.Parameter`` objects."""

    def __init__(self, problem, *, dtype, device):
        import torch

        self.names = problem.unknown_parameter_names
        self._values = [
            torch.nn.Parameter(
                torch.tensor(
                    problem.unknown_parameters[name].initial,
                    dtype=dtype,
                    device=device,
                )
            )
            for name in self.names
        ]

    def trainable(self):
        return list(self._values)

    def mapping(self):
        return dict(zip(self.names, self._values))

    def floats(self) -> dict[str, float]:
        return {
            name: float(value.detach().cpu())
            for name, value in zip(self.names, self._values)
        }

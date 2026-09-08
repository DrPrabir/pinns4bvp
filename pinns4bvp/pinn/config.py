"""Configuration objects for the forthcoming PyTorch PINN backend."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PINNConfig:
    """Training and network configuration for a PINN BVP solve.

    This configuration object is intentionally usable before the training
    backend is implemented, so the public API can be designed and tested first.
    """

    hidden_layers: tuple[int, ...] = (64, 64, 64)
    activation: str = "tanh"
    n_collocation: int = 200
    adam_epochs: int = 5000
    adam_lr: float = 1e-3
    use_lbfgs: bool = True
    lbfgs_max_iter: int = 500
    bc_weight: float = 10.0
    seed: int = 1234
    print_every: int = 500
    dtype: str = "float64"
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.hidden_layers or any(width < 1 for width in self.hidden_layers):
            raise ValueError("hidden_layers must contain positive integers")
        if self.activation.lower() not in {"tanh"}:
            raise ValueError("v0.2 scaffold currently reserves activation='tanh'")
        if self.n_collocation < 2:
            raise ValueError("n_collocation must be at least 2")
        if self.adam_epochs < 0:
            raise ValueError("adam_epochs must be >= 0")
        if self.adam_lr <= 0:
            raise ValueError("adam_lr must be > 0")
        if self.lbfgs_max_iter < 0:
            raise ValueError("lbfgs_max_iter must be >= 0")
        if self.bc_weight <= 0:
            raise ValueError("bc_weight must be > 0")
        if self.print_every < 1:
            raise ValueError("print_every must be >= 1")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'")

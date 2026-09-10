"""Configuration for the PyTorch PINN backend."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PINNConfig:
    """Network, optimizer, reproducibility, precision, and device settings.

    ``device`` may be ``"cpu"``, ``"cuda"``, ``"mps"``, or ``"auto"``.
    ``auto`` prefers CUDA; it uses MPS only for float32 and otherwise preserves
    float64 by falling back to CPU.
    """

    hidden_layers: tuple[int, ...] = (64, 64, 64)
    activation: str = "tanh"
    n_collocation: int = 200
    sampling: str = "uniform"

    ode_weight: float = 1.0
    bc_weight: float = 10.0

    adam_epochs: int = 5000
    adam_lr: float = 1e-3

    use_lbfgs: bool = True
    lbfgs_lr: float = 1.0
    lbfgs_max_iter: int = 500
    lbfgs_history_size: int = 100
    lbfgs_line_search: str | None = "strong_wolfe"

    seed: int = 1234
    deterministic: bool = True
    dtype: str = "float64"
    device: str = "cpu"

    loss_tolerance: float = 1e-10
    early_stopping_patience: int | None = 1000
    early_stopping_min_delta: float = 1e-12
    grad_clip_norm: float | None = None

    history_every: int = 10
    print_every: int = 500
    verbose: bool = True

    def __post_init__(self) -> None:
        if not self.hidden_layers or any(
            (not isinstance(width, int)) or width < 1 for width in self.hidden_layers
        ):
            raise ValueError("hidden_layers must contain positive integers")

        self.activation = self.activation.lower()
        if self.activation not in {"tanh", "sigmoid", "silu"}:
            raise ValueError("activation must be 'tanh', 'sigmoid', or 'silu'")

        self.sampling = self.sampling.lower()
        if self.sampling not in {"uniform", "random"}:
            raise ValueError("sampling must be 'uniform' or 'random'")

        if self.n_collocation < 2:
            raise ValueError("n_collocation must be at least 2")
        if self.ode_weight <= 0 or self.bc_weight <= 0:
            raise ValueError("ode_weight and bc_weight must be positive")

        if self.adam_epochs < 0:
            raise ValueError("adam_epochs must be >= 0")
        if self.adam_lr <= 0:
            raise ValueError("adam_lr must be positive")

        if self.lbfgs_lr <= 0:
            raise ValueError("lbfgs_lr must be positive")
        if self.lbfgs_max_iter < 0:
            raise ValueError("lbfgs_max_iter must be >= 0")
        if self.lbfgs_history_size < 1:
            raise ValueError("lbfgs_history_size must be >= 1")
        if self.lbfgs_line_search not in {None, "strong_wolfe"}:
            raise ValueError("lbfgs_line_search must be None or 'strong_wolfe'")

        if not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        self.dtype = self.dtype.lower()
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a non-empty string")
        self.device = self.device.lower().strip()
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'mps'")

        if self.loss_tolerance < 0:
            raise ValueError("loss_tolerance must be >= 0")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be >= 1 or None")
        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be >= 0")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be positive or None")
        if self.history_every < 1:
            raise ValueError("history_every must be >= 1")
        if self.print_every < 1:
            raise ValueError("print_every must be >= 1")

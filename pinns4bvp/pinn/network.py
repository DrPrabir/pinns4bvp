"""Neural-network definitions for the PyTorch PINN backend."""

from __future__ import annotations


class NormalizedMLP:
    """Factory-like wrapper retained for a clear public module boundary.

    Calling ``NormalizedMLP(...)`` returns a ``torch.nn.Module``.  PyTorch is
    imported lazily so the classical backend does not require PyTorch.
    """

    def __new__(
        cls,
        n_outputs: int,
        domain: tuple[float, float],
        *,
        hidden_layers: tuple[int, ...] = (64, 64, 64),
        activation: str = "tanh",
    ):
        import torch
        from torch import nn

        if n_outputs < 1:
            raise ValueError("n_outputs must be positive")
        a, b = map(float, domain)
        if not a < b:
            raise ValueError("domain must satisfy a < b")

        activation = activation.lower()
        activation_map = {
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "silu": nn.SiLU,
        }
        if activation not in activation_map:
            raise ValueError(f"unsupported activation '{activation}'")

        class _NormalizedMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("domain_a", torch.tensor(a))
                self.register_buffer("domain_b", torch.tensor(b))

                sizes = (1, *hidden_layers, n_outputs)
                layers: list[nn.Module] = []
                for i in range(len(sizes) - 2):
                    linear = nn.Linear(sizes[i], sizes[i + 1])
                    nn.init.xavier_normal_(linear.weight)
                    nn.init.zeros_(linear.bias)
                    layers.extend([linear, activation_map[activation]()])

                final = nn.Linear(sizes[-2], sizes[-1])
                nn.init.xavier_normal_(final.weight)
                nn.init.zeros_(final.bias)
                layers.append(final)
                self.net = nn.Sequential(*layers)

            def normalize(self, x):
                return 2.0 * (x - self.domain_a) / (self.domain_b - self.domain_a) - 1.0

            def forward(self, x):
                if x.ndim == 1:
                    x = x[:, None]
                if x.ndim != 2 or x.shape[1] != 1:
                    raise ValueError("x must have shape (N,) or (N, 1)")
                return self.net(self.normalize(x))

        return _NormalizedMLP()


def build_network(
    n_outputs: int,
    domain: tuple[float, float],
    *,
    hidden_layers: tuple[int, ...] = (64, 64, 64),
    activation: str = "tanh",
):
    """Build a domain-normalized fully connected PINN network."""

    return NormalizedMLP(
        n_outputs,
        domain,
        hidden_layers=hidden_layers,
        activation=activation,
    )

"""Neural-network definitions for the PyTorch PINN backend.

Implementation is intentionally deferred to the next v0.2 development step.
Keeping this file present now lets us stabilize module boundaries before adding
training behavior.
"""

from __future__ import annotations


def build_network(*args, **kwargs):
    """Build the PINN network.

    TODO(v0.2): implement a fully connected tanh network with one input and
    ``n_equations`` outputs, Xavier initialization, and domain normalization.
    """
    raise NotImplementedError(
        "PINN network construction is not implemented in the v0.2 scaffold yet."
    )

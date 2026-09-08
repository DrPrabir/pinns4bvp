from __future__ import annotations

import numpy as np


def create_initial_mesh(
    domain: tuple[float, float],
    n_nodes: int = 50,
    *,
    kind: str = "uniform",
) -> np.ndarray:
    """Create a validated initial mesh for the BVP solver."""
    a, b = map(float, domain)
    if not a < b:
        raise ValueError("domain must satisfy a < b")
    if not isinstance(n_nodes, int) or n_nodes < 5:
        raise ValueError("n_nodes must be an integer >= 5")

    if kind == "uniform":
        x = np.linspace(a, b, n_nodes)
    elif kind == "quadratic":
        s = np.linspace(0.0, 1.0, n_nodes)
        x = a + (b - a) * s**2
    else:
        raise ValueError("kind must be 'uniform' or 'quadratic'")

    return x


def validate_mesh(x: np.ndarray, domain: tuple[float, float]) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 5:
        raise ValueError("mesh must be a 1D array containing at least 5 nodes")
    if not np.all(np.isfinite(x)):
        raise ValueError("mesh must contain only finite values")
    if not np.all(np.diff(x) > 0):
        raise ValueError("mesh points must be strictly increasing")

    a, b = domain
    if not np.isclose(x[0], a) or not np.isclose(x[-1], b):
        raise ValueError("mesh endpoints must match the problem domain")
    return x

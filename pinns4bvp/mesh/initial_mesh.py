from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class MeshConfig:
    """Configuration for an initial BVP mesh.

    Parameters
    ----------
    n_nodes:
        Number of initial nodes. Must be at least 5.
    kind:
        ``uniform``, ``quadratic``/``left`` (cluster near the left endpoint),
        ``right`` (cluster near the right endpoint), or ``chebyshev`` (cluster
        near both endpoints).
    power:
        Clustering exponent for ``left`` and ``right`` meshes. Values greater
        than 1 increase clustering near the selected endpoint.
    """

    n_nodes: int = 50
    kind: str = "uniform"
    power: float = 2.0

    def __post_init__(self) -> None:
        if not isinstance(self.n_nodes, int) or self.n_nodes < 5:
            raise ValueError("n_nodes must be an integer >= 5")
        kind = self.kind.lower().strip()
        aliases = {"quadratic": "left", "quadratic-left": "left", "quadratic-right": "right"}
        kind = aliases.get(kind, kind)
        if kind not in {"uniform", "left", "right", "chebyshev"}:
            raise ValueError(
                "kind must be 'uniform', 'left'/'quadratic', 'right', or 'chebyshev'"
            )
        if not np.isfinite(self.power) or self.power <= 0:
            raise ValueError("power must be a positive finite number")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "power", float(self.power))


@dataclass(frozen=True, slots=True)
class MeshQualityReport:
    """Simple spacing diagnostics for a one-dimensional mesh."""

    n_nodes: int
    min_spacing: float
    max_spacing: float
    spacing_ratio: float

    def summary(self) -> str:
        return (
            "Mesh quality\n"
            "------------\n"
            f"Nodes          : {self.n_nodes}\n"
            f"Minimum spacing: {self.min_spacing:.3e}\n"
            f"Maximum spacing: {self.max_spacing:.3e}\n"
            f"Spacing ratio  : {self.spacing_ratio:.3e}"
        )


def create_initial_mesh(
    domain: tuple[float, float],
    n_nodes: int = 50,
    *,
    kind: str = "uniform",
    power: float = 2.0,
) -> np.ndarray:
    """Create a validated initial mesh for a two-point BVP.

    ``kind='quadratic'`` remains a backward-compatible alias for a left-clustered
    mesh with ``power=2``.
    """

    config = MeshConfig(n_nodes=n_nodes, kind=kind, power=power)
    a, b = map(float, domain)
    if not np.isfinite(a) or not np.isfinite(b) or not a < b:
        raise ValueError("domain must contain finite endpoints satisfying a < b")

    s = np.linspace(0.0, 1.0, config.n_nodes)
    if config.kind == "uniform":
        mapped = s
    elif config.kind == "left":
        mapped = s**config.power
    elif config.kind == "right":
        mapped = 1.0 - (1.0 - s) ** config.power
    else:  # chebyshev / cosine clustering at both endpoints
        mapped = 0.5 * (1.0 - np.cos(np.pi * s))

    x = a + (b - a) * mapped
    x[0] = a
    x[-1] = b
    return validate_mesh(x, (a, b))


def mesh_from_config(domain: tuple[float, float], config: MeshConfig) -> np.ndarray:
    """Create a mesh from :class:`MeshConfig`."""

    if not isinstance(config, MeshConfig):
        raise TypeError("config must be a MeshConfig")
    return create_initial_mesh(
        domain,
        config.n_nodes,
        kind=config.kind,
        power=config.power,
    )


def validate_mesh(x: np.ndarray, domain: tuple[float, float]) -> np.ndarray:
    """Validate and return a copy of a user-supplied one-dimensional mesh."""

    x = np.asarray(x, dtype=float).copy()
    if x.ndim != 1 or x.size < 5:
        raise ValueError("mesh must be a 1D array containing at least 5 nodes")
    if not np.all(np.isfinite(x)):
        raise ValueError("mesh must contain only finite values")
    if not np.all(np.diff(x) > 0):
        raise ValueError("mesh points must be strictly increasing")

    a, b = map(float, domain)
    if not np.isclose(x[0], a) or not np.isclose(x[-1], b):
        raise ValueError("mesh endpoints must match the problem domain")
    return x


def mesh_quality(x: np.ndarray, domain: tuple[float, float] | None = None) -> MeshQualityReport:
    """Return spacing diagnostics for a mesh."""

    arr = np.asarray(x, dtype=float)
    if domain is None:
        if arr.ndim != 1 or arr.size < 2:
            raise ValueError("mesh must be a 1D array containing at least 2 nodes")
        domain = (float(arr[0]), float(arr[-1]))
    arr = validate_mesh(arr, domain)
    dx = np.diff(arr)
    min_dx = float(np.min(dx))
    max_dx = float(np.max(dx))
    return MeshQualityReport(
        n_nodes=int(arr.size),
        min_spacing=min_dx,
        max_spacing=max_dx,
        spacing_ratio=max_dx / min_dx,
    )

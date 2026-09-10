"""Device discovery and selection for the PyTorch PINN backend."""

from __future__ import annotations

from dataclasses import dataclass


_VALID_DEVICES = {"auto", "cpu", "cuda", "mps"}


@dataclass(frozen=True, slots=True)
class DeviceAvailability:
    """Availability summary for PyTorch execution devices."""

    cpu: bool
    cuda: bool
    mps: bool

    def as_dict(self) -> dict[str, bool]:
        return {"cpu": self.cpu, "cuda": self.cuda, "mps": self.mps}

    def available(self) -> tuple[str, ...]:
        return tuple(name for name, ok in self.as_dict().items() if ok)


def device_availability() -> DeviceAvailability:
    """Return currently available PyTorch execution devices.

    PyTorch is imported lazily so classical-only installations do not require
    it until PINN functionality is used.
    """

    import torch

    cuda = bool(torch.cuda.is_available())
    mps = bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )
    return DeviceAvailability(cpu=True, cuda=cuda, mps=mps)


def available_devices() -> tuple[str, ...]:
    """Return available device names among ``cpu``, ``cuda``, and ``mps``."""

    return device_availability().available()


def resolve_device(requested: str, *, dtype: str = "float64") -> str:
    """Resolve a user device request to an available PyTorch device.

    Parameters
    ----------
    requested:
        One of ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"mps"``.
    dtype:
        ``"float32"`` or ``"float64"``.  Explicit MPS execution currently
        requires ``float32`` in PINNs4BVP because float64 support is not
        assumed across supported PyTorch/macOS combinations.

    Notes
    -----
    ``auto`` prefers CUDA when available.  It uses MPS only for float32.  For
    float64 on an Apple Silicon system without CUDA, ``auto`` deliberately
    falls back to CPU so the requested precision is preserved.
    """

    requested = str(requested).lower().strip()
    dtype = str(dtype).lower().strip()
    if requested not in _VALID_DEVICES:
        raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'mps'")
    if dtype not in {"float32", "float64"}:
        raise ValueError("dtype must be 'float32' or 'float64'")

    availability = device_availability()

    if requested == "cpu":
        return "cpu"

    if requested == "cuda":
        if not availability.cuda:
            raise RuntimeError(
                "CUDA was requested but is not available to the current PyTorch installation"
            )
        return "cuda"

    if requested == "mps":
        if not availability.mps:
            raise RuntimeError(
                "MPS was requested but is not available to the current PyTorch/macOS installation"
            )
        if dtype == "float64":
            raise ValueError(
                "PINNs4BVP does not assume float64 support on MPS. "
                "Use dtype='float32' with device='mps', or use device='cpu' "
                "for float64 training."
            )
        return "mps"

    # auto
    if availability.cuda:
        return "cuda"
    if availability.mps and dtype == "float32":
        return "mps"
    return "cpu"


def device_summary() -> str:
    """Return a short human-readable device availability summary."""

    info = device_availability()
    return (
        "PyTorch device availability\n"
        "---------------------------\n"
        f"CPU  : {info.cpu}\n"
        f"CUDA : {info.cuda}\n"
        f"MPS  : {info.mps}"
    )

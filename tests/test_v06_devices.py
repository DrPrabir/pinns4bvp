import pytest


torch = pytest.importorskip("torch")

from pinns4bvp.pinn import PINNConfig
from pinns4bvp.pinn.device import available_devices, device_availability, resolve_device


def test_device_config_and_cpu_resolution():
    assert PINNConfig(device="cpu").device == "cpu"
    assert PINNConfig(device="CUDA").device == "cuda"
    assert PINNConfig(device="MPS", dtype="float32").device == "mps"
    assert resolve_device("cpu", dtype="float64") == "cpu"
    with pytest.raises(ValueError):
        PINNConfig(device="tpu")


def test_device_availability_has_cpu_and_auto_resolves():
    info = device_availability()
    assert info.cpu is True
    assert "cpu" in available_devices()
    resolved64 = resolve_device("auto", dtype="float64")
    assert resolved64 in {"cpu", "cuda"}
    resolved32 = resolve_device("auto", dtype="float32")
    assert resolved32 in {"cpu", "cuda", "mps"}


def test_explicit_unavailable_accelerators_fail_clearly():
    info = device_availability()
    if not info.cuda:
        with pytest.raises(RuntimeError, match="CUDA"):
            resolve_device("cuda", dtype="float64")
    if not info.mps:
        with pytest.raises(RuntimeError, match="MPS"):
            resolve_device("mps", dtype="float32")
    else:
        with pytest.raises(ValueError, match="float64"):
            resolve_device("mps", dtype="float64")

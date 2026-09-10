"""v0.6 example: inspect and choose the PyTorch PINN execution device."""

from pinns4bvp.pinn import PINNConfig, device_summary, resolve_device


print(device_summary())
print()

for dtype in ("float32", "float64"):
    config = PINNConfig(device="auto", dtype=dtype, verbose=False)
    selected = resolve_device(config.device, dtype=config.dtype)
    print(f"auto with {dtype:<7} -> {selected}")

print("\nExamples:")
print("PINNConfig(device='cpu',  dtype='float64')")
print("PINNConfig(device='cuda', dtype='float64')  # when CUDA is available")
print("PINNConfig(device='mps',  dtype='float32')  # Apple Silicon / supported macOS")

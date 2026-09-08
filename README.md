# PINNs4BVP

PINNs4BVP is a general-purpose Python framework for two-point boundary-value
problems (BVPs).  It provides a classical SciPy collocation backend and is
developing an interchangeable PyTorch physics-informed neural-network backend.

## v0.3 development focus

The current development snapshot adds a robust PINN training engine:

- deterministic seeds and reproducibility controls;
- Adam optimization;
- optional L-BFGS refinement;
- ODE and boundary loss diagnostics;
- early stopping and loss tolerances;
- optional gradient clipping;
- structured training histories;
- backend-independent solution evaluation.

The core is application-neutral: it works with general first-order systems

`y' = f(x, y, p)`

and two-point boundary residuals.

## Install for development

```bash
python -m pip install -e ".[dev]"
```

## Classical solve

```python
sol = solve(problem, method="collocation")
```

## PINN solve (development API)

During the alpha releases, PINN problems provide PyTorch-native equation and
boundary callbacks in addition to the NumPy callbacks used by SciPy:

```python
config = PINNConfig(seed=1234)
sol = solve(problem, method="pinn", pinn_config=config)
```

See `pinns4bvp/examples/pinn_linear_bvp.py`.

## License

MIT.

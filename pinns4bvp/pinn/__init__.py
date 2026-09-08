"""PINN components for PINNs4BVP.

The v0.2 development branch introduces these modules incrementally.  The
classical SciPy backend remains the default until the PINN backend is complete
and validated.
"""

from pinns4bvp.pinn.config import PINNConfig

__all__ = ["PINNConfig"]

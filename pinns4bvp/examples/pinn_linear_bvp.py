"""Planned v0.2 PINN validation example.

Problem
-------
    y'' = -1,  0 <= x <= 1
    y(0) = 0
    y(1) = 0

Exact solution
--------------
    y(x) = x(1-x)/2

This file is intentionally a scaffold.  It will become the first end-to-end
PINN example after the network, autodiff, and PINN backend modules are
implemented and tested.
"""


def main() -> None:
    raise NotImplementedError(
        "pinn_linear_bvp is reserved for the v0.2 PINN implementation step."
    )


if __name__ == "__main__":
    main()

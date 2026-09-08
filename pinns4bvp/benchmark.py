"""Benchmarking utilities for comparing BVP solution backends."""

from __future__ import annotations


def compare_solutions(*args, **kwargs):
    """Compare two BVP solutions on a common grid.

    TODO(v0.2): report RMSE, MAE, maximum absolute error, and relative L2 error.
    """
    raise NotImplementedError(
        "Solution benchmarking is not implemented in the v0.2 scaffold yet."
    )

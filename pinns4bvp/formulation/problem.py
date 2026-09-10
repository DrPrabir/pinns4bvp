"""Public constructor for higher-order boundary-value problems."""

from __future__ import annotations

from pinns4bvp.formulation.compiler import compile_higher_order_bvp
from pinns4bvp.formulation.expressions import BoundaryCondition, Equation


def HigherOrderBVP(
    *,
    equation=None,
    equations=None,
    boundary_conditions,
    domain=(0.0, 1.0),
    name="Higher-order boundary-value problem",
    singular_matrix=None,
):
    """Construct a high-level BVP and compile it to :class:`BVPProblem`.

    Exactly one of ``equation=`` or ``equations=`` must be supplied.  The
    returned object is the established low-level :class:`BVPProblem`, so all
    v0.1-v0.7 solvers, benchmarking, diagnostics and continuation machinery
    remain available without a compatibility layer.
    """

    if (equation is None) == (equations is None):
        raise ValueError("supply exactly one of equation= or equations=")
    if equation is not None:
        equations = [equation]
    if isinstance(equations, Equation):
        equations = [equations]
    bcs = list(boundary_conditions)
    # Relations created with ``==`` are BoundaryCondition instances.
    if any(not isinstance(bc, BoundaryCondition) for bc in bcs):
        raise TypeError(
            "boundary_conditions must use high-level relations, for example y.at(0) == 0"
        )
    return compile_higher_order_bvp(
        equations=equations,
        boundary_conditions=bcs,
        domain=tuple(map(float, domain)),
        name=name,
        singular_matrix=singular_matrix,
    )


__all__ = ["HigherOrderBVP"]

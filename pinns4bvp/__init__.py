"""PINNs4BVP: general-purpose boundary-value problem tools."""

from pinns4bvp.benchmark import benchmark_problem
from pinns4bvp.continuation import ContinuationConfig, SolutionFamily, continue_parameter
from pinns4bvp.formulation import (
    BoundaryCondition,
    DependentVariable,
    Equation,
    HigherOrderBVP,
    IndependentVariable,
    Parameter,
    cos,
    d,
    d2,
    d3,
    d4,
    derivative,
    exp,
    log,
    sin,
    sqrt,
    tanh,
    x,
    y,
)
from pinns4bvp.parameters import Unknown, UnknownParameter
from pinns4bvp.problem import BVPProblem
from pinns4bvp.solution import BVPSolution
from pinns4bvp.solver import solve

__all__ = [
    "BVPProblem",
    "HigherOrderBVP",
    "IndependentVariable",
    "DependentVariable",
    "Parameter",
    "Equation",
    "BoundaryCondition",
    "x",
    "y",
    "derivative",
    "d",
    "d2",
    "d3",
    "d4",
    "exp",
    "sin",
    "cos",
    "tanh",
    "sqrt",
    "log",
    "ContinuationConfig",
    "BVPSolution",
    "Unknown",
    "UnknownParameter",
    "SolutionFamily",
    "benchmark_problem",
    "continue_parameter",
    "solve",
]
__version__ = "0.8.0.dev1"

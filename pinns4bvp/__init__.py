"""PINNs4BVP: general-purpose boundary-value problem tools."""

from pinns4bvp.benchmark import benchmark_problem
from pinns4bvp.continuation import ContinuationConfig, SolutionFamily, continue_parameter
from pinns4bvp.parameters import Unknown, UnknownParameter
from pinns4bvp.problem import BVPProblem
from pinns4bvp.solution import BVPSolution
from pinns4bvp.solver import solve

__all__ = [
    "BVPProblem",
    "ContinuationConfig",
    "BVPSolution",
    "Unknown",
    "UnknownParameter",
    "SolutionFamily",
    "benchmark_problem",
    "continue_parameter",
    "solve",
]
__version__ = "0.7.0.dev0"

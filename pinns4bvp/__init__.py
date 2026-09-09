"""PINNs4BVP: general-purpose boundary-value problem tools."""

from pinns4bvp.benchmark import benchmark_problem
from pinns4bvp.problem import BVPProblem
from pinns4bvp.solution import BVPSolution
from pinns4bvp.solver import solve

__all__ = ["BVPProblem", "BVPSolution", "benchmark_problem", "solve"]
__version__ = "0.4.0.dev0"

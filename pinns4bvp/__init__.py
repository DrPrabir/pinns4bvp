"""PINNs4BVP: general-purpose boundary-value problem tools."""

from pinns4bvp.problem import BVPProblem
from pinns4bvp.solution import BVPSolution
from pinns4bvp.solver import solve

__all__ = ["BVPProblem", "BVPSolution", "solve"]
__version__ = "0.3.0.dev0"

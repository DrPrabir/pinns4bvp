"""General benchmarking utilities for numerical, PINN, and exact BVP solutions."""

from pinns4bvp.benchmark.exact import ExactReference
from pinns4bvp.benchmark.metrics import (
    ComparisonMetrics,
    ParameterMetrics,
    compare_solutions,
    error_metrics,
    parameter_metrics,
)
from pinns4bvp.benchmark.report import BenchmarkReport, MethodRun
from pinns4bvp.benchmark.runner import benchmark_problem

__all__ = [
    "BenchmarkReport",
    "ComparisonMetrics",
    "ExactReference",
    "MethodRun",
    "ParameterMetrics",
    "benchmark_problem",
    "compare_solutions",
    "error_metrics",
    "parameter_metrics",
]

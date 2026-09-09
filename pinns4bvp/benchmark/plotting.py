"""Plotting helpers for benchmark reports."""

from __future__ import annotations


def plot_benchmark(report, *, variable=0, ax=None):
    import matplotlib.pyplot as plt

    idx = report.problem.variable_index(variable)
    name = report.problem.variable_names[idx]
    x = report.x

    if ax is None:
        _, ax = plt.subplots()

    if report.exact is not None and report.exact.has_variable(name):
        ax.plot(x, report.exact.values(name, x), label="Exact")
    if report.numerical is not None:
        ax.plot(x, report.numerical.solution.values(name, x), label="Numerical")
    if report.pinn is not None:
        ax.plot(x, report.pinn.solution.values(name, x), label="PINN")

    ax.set_xlabel("x")
    ax.set_ylabel(name)
    ax.set_title(f"Benchmark: {report.problem.name}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return ax

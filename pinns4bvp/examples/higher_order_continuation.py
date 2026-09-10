"""v0.8: high-level parameter continuation and storage policies."""

from pinns4bvp import Equation, HigherOrderBVP, Parameter, d2, y


lam = Parameter("lambda", value=0.0)
problem = HigherOrderBVP(
    equation=Equation(d2(y), 0.0),
    boundary_conditions=[y.at(0.0) == 0.0, y.at(1.0) == lam],
    domain=(0.0, 1.0),
)

family = problem.continue_parameter(
    lam,
    start=0.0,
    stop=2.0,
    step=0.25,
    save="requested",
    solve_kwargs={"tol": 1e-9},
)
print(family.summary())

# If continuation is only a route to one difficult target, keep only the end:
final = problem.continue_to(
    lam,
    target=2.0,
    start=0.0,
    step=0.25,
    solve_kwargs={"tol": 1e-9},
)
print("final lambda:", final.problem.parameters["lambda"])

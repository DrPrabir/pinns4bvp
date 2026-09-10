"""Compile higher-order expressions into the established first-order BVP API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from pinns4bvp.parameters import UnknownParameter
from pinns4bvp.problem import BVPProblem
from pinns4bvp.formulation.expressions import (
    AtExpression,
    BinaryExpression,
    BoundaryCondition,
    Constant,
    DependentVariable,
    DerivativeExpression,
    Equation,
    Expression,
    FunctionExpression,
    IndependentVariable,
    Parameter,
    UnaryExpression,
    walk_expression,
    x as default_x,
)


@dataclass(frozen=True, slots=True)
class StateEntry:
    variable: str
    derivative_order: int
    index: int
    name: str


@dataclass(frozen=True, slots=True)
class FormulationMetadata:
    independent_variable: str
    dependent_orders: dict[str, int]
    state_entries: tuple[StateEntry, ...]

    @property
    def state_index(self) -> dict[tuple[str, int], int]:
        return {(s.variable, s.derivative_order): s.index for s in self.state_entries}


def _state_name(variable: str, order: int, independent: str) -> str:
    if order == 0:
        return variable
    if order == 1:
        return f"d({variable})/d{independent}"
    return f"d{order}({variable})/d{independent}{order}"


def _collect_nodes(expressions: Iterable[Expression], cls):
    out = []
    for expr in expressions:
        out.extend(node for node in walk_expression(expr) if isinstance(node, cls))
    return out


def _dedupe_variables(nodes: Iterable[DependentVariable]) -> dict[str, DependentVariable]:
    result: dict[str, DependentVariable] = {}
    for node in nodes:
        old = result.get(node.name)
        if old is not None and old.order is not None and node.order is not None and old.order != node.order:
            raise ValueError(
                f"dependent variable '{node.name}' was declared with conflicting orders "
                f"{old.order} and {node.order}"
            )
        if old is None or (old.order is None and node.order is not None):
            result[node.name] = node
    return result


def _dedupe_parameters(nodes: Iterable[Parameter]) -> dict[str, Parameter]:
    result: dict[str, Parameter] = {}
    for node in nodes:
        old = result.get(node.name)
        if old is None:
            result[node.name] = node
            continue
        if (old.value, old.initial, old.unknown) != (node.value, node.initial, node.unknown):
            raise ValueError(f"parameter '{node.name}' was declared inconsistently")
    return result


def _zero():
    return Constant(0.0)


def _one():
    return Constant(1.0)


def _combine_coeffs(a, b, op):
    keys = set(a) | set(b)
    out = {}
    for key in keys:
        left = a.get(key, _zero())
        right = b.get(key, _zero())
        out[key] = left + right if op == "add" else left - right
    return out


def affine_decompose(expr: Expression, targets: set[tuple[str, int]]):
    """Return ``(coefficients, rest)`` for an expression affine in targets.

    Each target is a highest derivative ``(variable_name, derivative_order)``.
    A clear error is raised when highest derivatives occur nonlinearly.
    """

    if isinstance(expr, DerivativeExpression) and expr.key in targets:
        return {expr.key: _one()}, _zero()

    if isinstance(expr, (Constant, IndependentVariable, DependentVariable, Parameter, DerivativeExpression)):
        return {}, expr

    if isinstance(expr, UnaryExpression):
        coeff, rest = affine_decompose(expr.arg, targets)
        if expr.op != "neg":
            raise ValueError(f"unsupported unary expression '{expr.op}'")
        return {k: -v for k, v in coeff.items()}, -rest

    if isinstance(expr, BinaryExpression):
        lc, lr = affine_decompose(expr.left, targets)
        rc, rr = affine_decompose(expr.right, targets)
        if expr.op == "add":
            return _combine_coeffs(lc, rc, "add"), lr + rr
        if expr.op == "sub":
            return _combine_coeffs(lc, rc, "sub"), lr - rr
        if expr.op == "mul":
            if lc and rc:
                raise ValueError(
                    "highest derivatives must enter the higher-order equations linearly; "
                    "products of highest derivatives are not supported"
                )
            if lc:
                return {k: v * rr for k, v in lc.items()}, lr * rr
            if rc:
                return {k: v * lr for k, v in rc.items()}, lr * rr
            return {}, lr * rr
        if expr.op == "div":
            if rc:
                raise ValueError("a highest derivative cannot appear in a denominator")
            return {k: v / rr for k, v in lc.items()}, lr / rr
        if expr.op == "pow":
            if lc or rc:
                raise ValueError(
                    "highest derivatives must enter linearly and cannot appear inside powers"
                )
            return {}, lr ** rr
        raise ValueError(f"unsupported binary expression '{expr.op}'")

    if isinstance(expr, FunctionExpression):
        coeff, rest = affine_decompose(expr.arg, targets)
        if coeff:
            raise ValueError(
                f"highest derivatives cannot appear inside {expr.name}() in v0.8"
            )
        return {}, FunctionExpression(expr.name, rest)

    if isinstance(expr, AtExpression):
        raise ValueError(".at(...) is only valid in boundary conditions")

    raise TypeError(f"unsupported expression node {type(expr).__name__}")


def _backend_function(name, value, backend):
    if backend == "numpy":
        fn = getattr(np, name)
        return fn(value)
    import torch

    fn = getattr(torch, name)
    return fn(value)


def _evaluate(expr, *, coordinate, state, parameters, state_index, backend):
    if isinstance(expr, Constant):
        return expr.value
    if isinstance(expr, IndependentVariable):
        return coordinate
    if isinstance(expr, Parameter):
        return parameters[expr.name]
    if isinstance(expr, DependentVariable):
        return state[state_index[(expr.name, 0)]]
    if isinstance(expr, DerivativeExpression):
        key = (expr.variable.name, expr.order)
        if key not in state_index:
            raise ValueError(
                f"derivative {expr} is not an internal state; highest derivatives must be "
                "eliminated by the equation compiler"
            )
        return state[state_index[key]]
    if isinstance(expr, UnaryExpression):
        value = _evaluate(
            expr.arg,
            coordinate=coordinate,
            state=state,
            parameters=parameters,
            state_index=state_index,
            backend=backend,
        )
        if expr.op == "neg":
            return -value
        raise ValueError(f"unsupported unary operator {expr.op}")
    if isinstance(expr, BinaryExpression):
        left = _evaluate(
            expr.left,
            coordinate=coordinate,
            state=state,
            parameters=parameters,
            state_index=state_index,
            backend=backend,
        )
        right = _evaluate(
            expr.right,
            coordinate=coordinate,
            state=state,
            parameters=parameters,
            state_index=state_index,
            backend=backend,
        )
        if expr.op == "add":
            return left + right
        if expr.op == "sub":
            return left - right
        if expr.op == "mul":
            return left * right
        if expr.op == "div":
            return left / right
        if expr.op == "pow":
            return left ** right
        raise ValueError(f"unsupported binary operator {expr.op}")
    if isinstance(expr, FunctionExpression):
        value = _evaluate(
            expr.arg,
            coordinate=coordinate,
            state=state,
            parameters=parameters,
            state_index=state_index,
            backend=backend,
        )
        return _backend_function(expr.name, value, backend)
    raise TypeError(f"cannot evaluate {type(expr).__name__} as an ODE expression")


def _as_vector(value, coordinate, backend):
    if backend == "numpy":
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return np.full_like(np.asarray(coordinate, dtype=float), float(arr), dtype=float)
        return np.broadcast_to(arr, np.asarray(coordinate).shape)
    import torch

    if torch.is_tensor(value):
        if value.ndim == 0:
            return torch.ones_like(coordinate) * value
        return torch.broadcast_to(value, coordinate.shape)
    return torch.ones_like(coordinate) * float(value)


def _solve_highest(coeff_rows, rest_rows, *, coordinate, state, parameters, state_index, backend):
    m = len(coeff_rows)
    columns = []
    for row in coeff_rows:
        columns.append([
            _as_vector(
                _evaluate(
                    expr,
                    coordinate=coordinate,
                    state=state,
                    parameters=parameters,
                    state_index=state_index,
                    backend=backend,
                ),
                coordinate,
                backend,
            )
            for expr in row
        ])
    rhs = [
        -_as_vector(
            _evaluate(
                expr,
                coordinate=coordinate,
                state=state,
                parameters=parameters,
                state_index=state_index,
                backend=backend,
            ),
            coordinate,
            backend,
        )
        for expr in rest_rows
    ]

    if backend == "numpy":
        # A[n, equation, highest-derivative]
        A = np.stack([np.stack(row, axis=-1) for row in columns], axis=-2)
        b = np.stack(rhs, axis=-1)
        try:
            solved = np.linalg.solve(A, b[..., None])[..., 0]
        except np.linalg.LinAlgError as exc:
            raise ValueError("highest-derivative coefficient matrix is singular") from exc
        return solved.T

    import torch

    A = torch.stack([torch.stack(row, dim=-1) for row in columns], dim=-2)
    b = torch.stack(rhs, dim=-1)
    try:
        solved = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)
    except RuntimeError as exc:
        raise ValueError("highest-derivative coefficient matrix is singular") from exc
    return solved.T


def _endpoint_side(location: float, domain: tuple[float, float], *, atol=1e-12):
    a, b = domain
    if abs(location - a) <= atol:
        return "left"
    if abs(location - b) <= atol:
        return "right"
    raise ValueError(
        f"two-point boundary conditions must use domain endpoints {domain}; got x={location}"
    )


def _eval_boundary_expr(expr, *, ya, yb, parameters, state_index, domain, backend):
    if isinstance(expr, AtExpression):
        side = _endpoint_side(expr.location, domain)
        state = ya if side == "left" else yb
        coordinate = domain[0] if side == "left" else domain[1]
        return _evaluate(
            expr.expression,
            coordinate=coordinate,
            state=state,
            parameters=parameters,
            state_index=state_index,
            backend=backend,
        )
    if isinstance(expr, Constant):
        return expr.value
    if isinstance(expr, Parameter):
        return parameters[expr.name]
    if isinstance(expr, UnaryExpression):
        value = _eval_boundary_expr(
            expr.arg,
            ya=ya,
            yb=yb,
            parameters=parameters,
            state_index=state_index,
            domain=domain,
            backend=backend,
        )
        if expr.op == "neg":
            return -value
    if isinstance(expr, BinaryExpression):
        left = _eval_boundary_expr(
            expr.left,
            ya=ya,
            yb=yb,
            parameters=parameters,
            state_index=state_index,
            domain=domain,
            backend=backend,
        )
        right = _eval_boundary_expr(
            expr.right,
            ya=ya,
            yb=yb,
            parameters=parameters,
            state_index=state_index,
            domain=domain,
            backend=backend,
        )
        if expr.op == "add": return left + right
        if expr.op == "sub": return left - right
        if expr.op == "mul": return left * right
        if expr.op == "div": return left / right
        if expr.op == "pow": return left ** right
    if isinstance(expr, FunctionExpression):
        value = _eval_boundary_expr(
            expr.arg,
            ya=ya,
            yb=yb,
            parameters=parameters,
            state_index=state_index,
            domain=domain,
            backend=backend,
        )
        return _backend_function(expr.name, value, backend)
    if isinstance(expr, (DependentVariable, DerivativeExpression, IndependentVariable)):
        raise ValueError(
            "state variables and derivatives in boundary conditions must be located with .at(endpoint)"
        )
    raise TypeError(f"unsupported boundary expression {type(expr).__name__}")


def compile_higher_order_bvp(
    *,
    equations: Iterable[Equation],
    boundary_conditions: Iterable[BoundaryCondition],
    domain: tuple[float, float],
    name: str = "Higher-order boundary-value problem",
    singular_matrix=None,
) -> BVPProblem:
    equations = tuple(equations)
    bcs = tuple(boundary_conditions)
    if not equations:
        raise ValueError("at least one higher-order equation is required")
    if not bcs:
        raise ValueError("at least one boundary condition is required")
    if any(not isinstance(eq, Equation) for eq in equations):
        raise TypeError("equations must contain Equation objects")
    if any(not isinstance(bc, BoundaryCondition) for bc in bcs):
        raise TypeError("boundary_conditions must contain relations such as y.at(0) == 0")

    ode_exprs = [eq.residual for eq in equations]
    bc_exprs = [bc.residual for bc in bcs]
    all_exprs = ode_exprs + bc_exprs

    independent_nodes = _collect_nodes(all_exprs, IndependentVariable)
    independent_names = {node.name for node in independent_nodes}
    if len(independent_names) > 1:
        raise ValueError(
            f"a BVP may use only one independent variable; found {sorted(independent_names)}"
        )
    independent = independent_nodes[0] if independent_nodes else default_x

    variables = _dedupe_variables(_collect_nodes(all_exprs, DependentVariable))
    if not variables:
        raise ValueError("no dependent variable was found in the equations")

    derivative_nodes = _collect_nodes(all_exprs, DerivativeExpression)
    inferred_orders = {name: 0 for name in variables}
    for node in derivative_nodes:
        if node.independent.name != independent.name:
            raise ValueError("all derivatives must use the same independent variable")
        inferred_orders[node.variable.name] = max(
            inferred_orders.get(node.variable.name, 0), node.order
        )

    orders: dict[str, int] = {}
    for name_, variable in variables.items():
        inferred = inferred_orders.get(name_, 0)
        if variable.order is None:
            if inferred < 1:
                raise ValueError(
                    f"cannot infer differential order for '{name_}'; declare order= explicitly"
                )
            orders[name_] = inferred
        else:
            if inferred > variable.order:
                raise ValueError(
                    f"derivative order {inferred} exceeds declared order={variable.order} "
                    f"for '{name_}'"
                )
            orders[name_] = variable.order

    if len(equations) != len(orders):
        raise ValueError(
            "the explicit higher-order formulation requires one governing equation per "
            f"dependent variable; found {len(equations)} equations for {len(orders)} variables"
        )

    parameter_nodes = _dedupe_parameters(_collect_nodes(all_exprs, Parameter))
    fixed = {
        name_: spec.value
        for name_, spec in parameter_nodes.items()
        if not spec.unknown
    }
    unknown = {
        name_: UnknownParameter(spec.initial, spec.description)
        for name_, spec in parameter_nodes.items()
        if spec.unknown
    }

    entries = []
    index = 0
    for name_, order in orders.items():
        for derivative_order in range(order):
            entries.append(
                StateEntry(
                    variable=name_,
                    derivative_order=derivative_order,
                    index=index,
                    name=_state_name(name_, derivative_order, independent.name),
                )
            )
            index += 1
    metadata = FormulationMetadata(
        independent_variable=independent.name,
        dependent_orders=dict(orders),
        state_entries=tuple(entries),
    )
    state_index = metadata.state_index

    targets = {(name_, order) for name_, order in orders.items()}
    target_order = [(name_, orders[name_]) for name_ in orders]
    coeff_rows = []
    rest_rows = []
    for expr in ode_exprs:
        coeffs, rest = affine_decompose(expr, targets)
        coeff_rows.append([coeffs.get(target, _zero()) for target in target_order])
        rest_rows.append(rest)
    for target in target_order:
        if not any(target in affine_decompose(expr, targets)[0] for expr in ode_exprs):
            raise ValueError(
                f"highest derivative d^{target[1]}({target[0]}) is not determined by the equations"
            )

    if len(bcs) != len(entries) + len(unknown):
        raise ValueError(
            "incorrect number of boundary conditions: the compiled first-order system has "
            f"{len(entries)} states and {len(unknown)} unknown parameters, so exactly "
            f"{len(entries) + len(unknown)} boundary conditions are required; got {len(bcs)}"
        )

    # Validate two-point locations and ensure derivative boundary values map to
    # retained first-order states.
    for expr in bc_exprs:
        for node in walk_expression(expr):
            if isinstance(node, AtExpression):
                _endpoint_side(node.location, domain)
                for child in walk_expression(node.expression):
                    if isinstance(child, DerivativeExpression):
                        order = orders[child.variable.name]
                        if child.order >= order:
                            raise ValueError(
                                "boundary conditions on the highest derivative are not supported "
                                f"for '{child.variable.name}' in the v0.8 high-level API"
                            )

    def equations_numpy(coord, state, p):
        n = np.asarray(coord).size
        out = np.empty((len(entries), n), dtype=float)
        for entry in entries:
            if entry.derivative_order < orders[entry.variable] - 1:
                out[entry.index] = state[state_index[(entry.variable, entry.derivative_order + 1)]]
        highest = _solve_highest(
            coeff_rows,
            rest_rows,
            coordinate=coord,
            state=state,
            parameters=p,
            state_index=state_index,
            backend="numpy",
        )
        for row, variable_name in enumerate(orders):
            last = state_index[(variable_name, orders[variable_name] - 1)]
            out[last] = highest[row]
        return out

    def equations_torch(coord, state, p):
        import torch

        rows = [None] * len(entries)
        for entry in entries:
            if entry.derivative_order < orders[entry.variable] - 1:
                rows[entry.index] = state[state_index[(entry.variable, entry.derivative_order + 1)]]
        highest = _solve_highest(
            coeff_rows,
            rest_rows,
            coordinate=coord,
            state=state,
            parameters=p,
            state_index=state_index,
            backend="torch",
        )
        for row, variable_name in enumerate(orders):
            last = state_index[(variable_name, orders[variable_name] - 1)]
            rows[last] = highest[row]
        return torch.stack(rows)

    def bc_numpy(ya, yb, p):
        return np.asarray([
            _eval_boundary_expr(
                expr,
                ya=ya,
                yb=yb,
                parameters=p,
                state_index=state_index,
                domain=domain,
                backend="numpy",
            )
            for expr in bc_exprs
        ], dtype=float)

    def bc_torch(ya, yb, p):
        import torch

        values = [
            _eval_boundary_expr(
                expr,
                ya=ya,
                yb=yb,
                parameters=p,
                state_index=state_index,
                domain=domain,
                backend="torch",
            )
            for expr in bc_exprs
        ]
        normalized = [
            value if torch.is_tensor(value) else torch.as_tensor(value, dtype=ya.dtype, device=ya.device)
            for value in values
        ]
        return torch.stack(normalized)

    return BVPProblem(
        equations=equations_numpy,
        boundary_conditions=bc_numpy,
        pinn_equations=equations_torch,
        pinn_boundary_conditions=bc_torch,
        domain=domain,
        n_equations=len(entries),
        parameters=fixed,
        unknown_parameters=unknown,
        variable_names=tuple(entry.name for entry in entries),
        name=name,
        singular_matrix=singular_matrix,
        formulation_metadata=metadata,
    )


__all__ = [
    "StateEntry",
    "FormulationMetadata",
    "affine_decompose",
    "compile_higher_order_bvp",
]

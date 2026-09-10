"""Structured mathematical expressions for the v0.8 higher-order BVP API."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Iterable


class Expression:
    """Base class for higher-order mathematical expressions."""

    __array_priority__ = 1000

    def __add__(self, other):
        return BinaryExpression("add", self, as_expression(other))

    def __radd__(self, other):
        return BinaryExpression("add", as_expression(other), self)

    def __sub__(self, other):
        return BinaryExpression("sub", self, as_expression(other))

    def __rsub__(self, other):
        return BinaryExpression("sub", as_expression(other), self)

    def __mul__(self, other):
        return BinaryExpression("mul", self, as_expression(other))

    def __rmul__(self, other):
        return BinaryExpression("mul", as_expression(other), self)

    def __truediv__(self, other):
        return BinaryExpression("div", self, as_expression(other))

    def __rtruediv__(self, other):
        return BinaryExpression("div", as_expression(other), self)

    def __pow__(self, other):
        return BinaryExpression("pow", self, as_expression(other))

    def __rpow__(self, other):
        return BinaryExpression("pow", as_expression(other), self)

    def __neg__(self):
        return UnaryExpression("neg", self)

    def __pos__(self):
        return self

    def __eq__(self, other):  # type: ignore[override]
        return BoundaryCondition(self, as_expression(other))

    def at(self, location: float):
        return AtExpression(self, float(location))


@dataclass(frozen=True, slots=True, eq=False)
class Constant(Expression):
    value: float

    def __post_init__(self):
        object.__setattr__(self, "value", float(self.value))


@dataclass(frozen=True, slots=True, eq=False)
class IndependentVariable(Expression):
    name: str = "x"

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("independent-variable name must be a non-empty string")

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True, eq=False)
class DependentVariable(Expression):
    """Dependent variable declaration.

    ``order=None`` lets PINNs4BVP infer the differential order from the
    highest derivative that appears in the equations.  For simultaneous or
    mixed-order systems, specifying ``order`` explicitly is recommended.
    """

    name: str = "y"
    order: int | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("dependent-variable name must be a non-empty string")
        if self.order is not None and (
            not isinstance(self.order, int) or self.order < 1
        ):
            raise ValueError("DependentVariable.order must be a positive integer or None")

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True, eq=False)
class Parameter(Expression):
    """Known or unknown scalar parameter for the high-level formulation API.

    Examples
    --------
    ``Parameter("a", value=2.0)`` declares a known parameter.
    ``Parameter("k", initial=3.0, unknown=True)`` declares a solved parameter.
    Supplying ``initial`` without ``value`` also implies ``unknown=True``.
    """

    name: str
    value: float | None = None
    initial: float | None = None
    unknown: bool = False
    description: str | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("parameter name must be a non-empty string")
        if self.value is not None and self.initial is not None:
            raise ValueError("Parameter accepts either value= or initial=, not both")
        inferred_unknown = bool(self.unknown or (self.value is None and self.initial is not None))
        object.__setattr__(self, "unknown", inferred_unknown)
        if inferred_unknown:
            if self.initial is None:
                raise ValueError("unknown Parameter requires initial=")
            object.__setattr__(self, "initial", float(self.initial))
            if self.value is not None:
                raise ValueError("unknown Parameter cannot also have value=")
        else:
            if self.value is None:
                raise ValueError("known Parameter requires value=")
            object.__setattr__(self, "value", float(self.value))
        if self.description is not None and not isinstance(self.description, str):
            raise TypeError("Parameter.description must be a string or None")

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True, eq=False)
class DerivativeExpression(Expression):
    variable: DependentVariable
    independent: IndependentVariable
    order: int

    def __post_init__(self):
        if not isinstance(self.variable, DependentVariable):
            raise TypeError("derivative variable must be a DependentVariable")
        if not isinstance(self.independent, IndependentVariable):
            raise TypeError("independent must be an IndependentVariable")
        if not isinstance(self.order, int) or self.order < 1:
            raise ValueError("derivative order must be a positive integer")

    @property
    def key(self) -> tuple[str, int]:
        return (self.variable.name, self.order)

    def __repr__(self) -> str:
        return f"d{self.order}({self.variable.name})/d{self.independent.name}{self.order}"


@dataclass(frozen=True, slots=True, eq=False)
class BinaryExpression(Expression):
    op: str
    left: Expression
    right: Expression


@dataclass(frozen=True, slots=True, eq=False)
class UnaryExpression(Expression):
    op: str
    arg: Expression


@dataclass(frozen=True, slots=True, eq=False)
class FunctionExpression(Expression):
    name: str
    arg: Expression


@dataclass(frozen=True, slots=True, eq=False)
class AtExpression(Expression):
    expression: Expression
    location: float


@dataclass(frozen=True, slots=True, eq=False)
class Equation:
    lhs: Expression
    rhs: Expression

    def __init__(self, lhs, rhs=0.0):
        object.__setattr__(self, "lhs", as_expression(lhs))
        object.__setattr__(self, "rhs", as_expression(rhs))

    @property
    def residual(self) -> Expression:
        return self.lhs - self.rhs


@dataclass(frozen=True, slots=True, eq=False)
class BoundaryCondition:
    lhs: Expression
    rhs: Expression

    def __init__(self, lhs, rhs=0.0):
        object.__setattr__(self, "lhs", as_expression(lhs))
        object.__setattr__(self, "rhs", as_expression(rhs))

    @property
    def residual(self) -> Expression:
        return self.lhs - self.rhs


def as_expression(value) -> Expression:
    if isinstance(value, Expression):
        return value
    if isinstance(value, Real):
        return Constant(float(value))
    raise TypeError(f"cannot convert {type(value).__name__} to a PINNs4BVP expression")


# Default mathematical symbols.  Users may explicitly declare alternatives.
x = IndependentVariable("x")
y = DependentVariable("y", order=None)


def derivative(variable, independent=None, *, order: int = 1):
    if not isinstance(variable, DependentVariable):
        raise TypeError("derivative() expects a DependentVariable")
    independent = x if independent is None else independent
    return DerivativeExpression(variable, independent, order)


def d(variable, independent=None):
    return derivative(variable, independent, order=1)


def d2(variable, independent=None):
    return derivative(variable, independent, order=2)


def d3(variable, independent=None):
    return derivative(variable, independent, order=3)


def d4(variable, independent=None):
    return derivative(variable, independent, order=4)


def _func(name: str, arg):
    return FunctionExpression(name, as_expression(arg))


def exp(arg):
    return _func("exp", arg)


def sin(arg):
    return _func("sin", arg)


def cos(arg):
    return _func("cos", arg)


def tanh(arg):
    return _func("tanh", arg)


def sqrt(arg):
    return _func("sqrt", arg)


def log(arg):
    return _func("log", arg)


def walk_expression(expr: Expression) -> Iterable[Expression]:
    yield expr
    if isinstance(expr, BinaryExpression):
        yield from walk_expression(expr.left)
        yield from walk_expression(expr.right)
    elif isinstance(expr, UnaryExpression):
        yield from walk_expression(expr.arg)
    elif isinstance(expr, FunctionExpression):
        yield from walk_expression(expr.arg)
    elif isinstance(expr, DerivativeExpression):
        yield expr.variable
        yield expr.independent
    elif isinstance(expr, AtExpression):
        yield from walk_expression(expr.expression)


__all__ = [
    "Expression",
    "Constant",
    "IndependentVariable",
    "DependentVariable",
    "Parameter",
    "DerivativeExpression",
    "Equation",
    "BoundaryCondition",
    "AtExpression",
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
    "walk_expression",
]

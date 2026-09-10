import numpy as np

from pinns4bvp import BVPProblem, solve


def test_optional_classical_jacobians_are_forwarded():
    calls = {"fun": 0, "bc": 0}

    def fun(x, y, p):
        return np.vstack((y[1], -y[0]))

    def bc(ya, yb, p):
        return np.array((ya[0], yb[0] - np.sin(1.0)))

    def fun_jac(x, y, p):
        calls["fun"] += 1
        m = x.size
        J = np.zeros((2, 2, m))
        J[0, 1, :] = 1.0
        J[1, 0, :] = -1.0
        return J

    def bc_jac(ya, yb, p):
        calls["bc"] += 1
        A = np.zeros((2, 2))
        B = np.zeros((2, 2))
        A[0, 0] = 1.0
        B[1, 0] = 1.0
        return A, B

    problem = BVPProblem(
        fun,
        bc,
        (0.0, 1.0),
        2,
        variable_names=("y", "yp"),
        equations_jacobian=fun_jac,
        boundary_jacobian=bc_jac,
    )
    sol = solve(problem, tol=1e-9)
    assert sol.success
    assert calls["fun"] > 0
    assert calls["bc"] > 0


def test_singular_matrix_is_validated():
    def fun(x, y, p):
        return np.zeros_like(y)

    def bc(ya, yb, p):
        return np.array((yb[0] - 1.0,))

    problem = BVPProblem(
        fun,
        bc,
        (0.0, 1.0),
        1,
        singular_matrix=np.array([[0.0]]),
    )
    assert problem.singular_matrix.shape == (1, 1)

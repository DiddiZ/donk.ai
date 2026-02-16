from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.testing import assert_allclose

if TYPE_CHECKING:
    from sympy import Expr, Symbol


def _create_symbols(D: int) -> tuple[tuple[Symbol, ...], tuple[Symbol, ...]]:
    from sympy import symbols

    d_sym = symbols(f"d0:{D}")
    wp_sym = symbols(f"w0:{D}")

    return d_sym, wp_sym


def _evaluate_loss(
    d: np.ndarray, wp: np.ndarray, d_sym: tuple[Symbol, ...], wp_sym: tuple[Symbol, ...], loss: Expr
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sympy import Matrix, diff

    T, D = d.shape

    # Derivatives
    d1 = Matrix([diff(loss, s) for s in d_sym])
    d2 = Matrix([[diff(d, s) for s in d_sym] for d in d1])

    l = np.empty((T,))
    lx = np.empty((T, D))
    lxx = np.empty((T, D, D))
    for t in range(T):  # Evaluate by substituting actual values
        substitutions = {d_sym[i]: d[t, i] for i in range(D)}
        substitutions.update({wp_sym[i]: wp[t, i] for i in range(D)})

        l[t] = float(loss.subs(substitutions))
        lx[t] = np.array(d1.subs(substitutions), dtype=float).reshape(D)
        lxx[t] = np.array(d2.subs(substitutions), dtype=float).reshape(D, D)
    return l, lx, lxx


def loss_l2_ref(x: np.ndarray, t: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute reference for loss_l2 using sympy for symbolic differentiation."""
    _, dX = x.shape
    d_sym, wp_sym = _create_symbols(dX)

    # Loss function
    loss = 0.5 * sum(d_sym[i] ** 2 * wp_sym[i] for i in range(dX))

    return _evaluate_loss(x - t, w, d_sym, wp_sym, loss)


def loss_l1_ref(x: np.ndarray, t: np.ndarray, w: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute reference for loss_l1 using sympy for symbolic differentiation."""
    from sympy import sqrt

    _, dX = x.shape
    d_sym, wp_sym = _create_symbols(dX)

    # Loss function
    loss = sum(sqrt(d_sym[i] ** 2 + alpha) * wp_sym[i] for i in range(dX))

    return _evaluate_loss(x - t, w, d_sym, wp_sym, loss)


def loss_log_cosh_ref(x: np.ndarray, t: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute reference for loss_log_cosh using sympy for symbolic differentiation."""
    from sympy import cosh, log

    _, dX = x.shape
    d_sym, wp_sym = _create_symbols(dX)

    # Loss function
    loss = sum(log(cosh(d_sym[i])) * wp_sym[i] for i in range(dX))

    return _evaluate_loss(x - t, w, d_sym, wp_sym, loss)


@pytest.mark.parametrize("seed", range(10))
def test_loss_l2(seed: int) -> None:
    """Test loss_l2 implementation agains reference implementation using random values."""
    from donk.costs import loss_l2

    T, dX = 10, 3
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((T, dX))
    t = rng.standard_normal((T, dX))
    w = rng.uniform(size=(T, dX))

    l_ref, lx_ref, lxx_ref = loss_l2_ref(x, t, w)
    l, lx, lxx = loss_l2(x, t, w)

    assert_allclose(l_ref, l)
    assert_allclose(lx_ref, lx)
    assert_allclose(lxx_ref, lxx)


@pytest.mark.parametrize("seed", range(10))
def test_loss_l1(seed: int) -> None:
    """Test loss_l1 implementation agains reference implementation using random values."""
    from donk.costs import loss_l1

    T, dX = 10, 3
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((T, dX))
    t = rng.standard_normal((T, dX))
    w = rng.uniform(size=(T, dX))

    l_ref, lx_ref, lxx_ref = loss_l1_ref(x, t, w, alpha=1e-2)
    l, lx, lxx = loss_l1(x, t, w, alpha=1e-2)

    assert_allclose(l_ref, l)
    assert_allclose(lx_ref, lx)
    assert_allclose(lxx_ref, lxx)


@pytest.mark.parametrize("seed", range(10))
def test_loss_log_cosh(seed: int) -> None:
    """Test loss_log_cosh implementation agains reference implementation using random values."""
    from donk.costs import loss_log_cosh

    T, dX = 10, 3
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((T, dX))
    t = rng.standard_normal((T, dX))
    w = rng.uniform(size=(T, dX))

    l_ref, lx_ref, lxx_ref = loss_log_cosh_ref(x, t, w)
    l, lx, lxx = loss_log_cosh(x, t, w)

    assert_allclose(l_ref, l)
    assert_allclose(lx_ref, lx)
    assert_allclose(lxx_ref, lxx)


@pytest.mark.parametrize("seed", range(10))
def test_loss_sum(seed: int) -> None:
    """Test loss_combined summing up two losses."""
    from donk.costs import loss_combined, loss_l2, loss_log_cosh

    T, dX = 10, 3
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((T, dX))
    t = rng.standard_normal((T, dX))
    w = rng.uniform(size=(T, dX))

    # Reference
    l_l2_ref, lx_l2_ref, lxx_l2_ref = loss_l2(x, t, w)
    l_log_cosh_ref, lx_log_cosh_ref, lxx_log_cosh_ref = loss_log_cosh_ref(x, t, w)
    l_ref = l_l2_ref + l_log_cosh_ref
    lx_ref = lx_l2_ref + lx_log_cosh_ref
    lxx_ref = lxx_l2_ref + lxx_log_cosh_ref

    # Impl
    l, lx, lxx = loss_combined(
        x,
        losses=[
            (loss_l2, {"t": t, "w": w}),
            (loss_log_cosh, {"t": t, "w": w}),
        ],
    )

    assert_allclose(l_ref, l)
    assert_allclose(lx_ref, lx)
    assert_allclose(lxx_ref, lxx)


@pytest.mark.parametrize("seed", range(10))
def test_loss_sum_single_argument(seed: int) -> None:
    """Test loss_combined with only a single loss to sum up."""
    from donk.costs import loss_combined, loss_l2

    T, dX = 10, 3
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((T, dX))
    t = rng.standard_normal((T, dX))
    w = rng.uniform(size=(T, dX))

    # Reference
    l_ref, lx_ref, lxx_ref = loss_l2_ref(x, t, w)

    # Impl
    l, lx, lxx = loss_combined(
        x,
        losses=[
            (loss_l2, {"t": t, "w": w}),
        ],
    )

    assert_allclose(l_ref, l)
    assert_allclose(lx_ref, lx)
    assert_allclose(lxx_ref, lxx)


def test_loss_sum_missing_arguments() -> None:
    """Test loss_combined raises errors on missing arguments."""
    from donk.costs import loss_combined

    T, dX = 10, 3
    rng = np.random.default_rng(0)
    x = rng.standard_normal((T, dX))

    # No losses to sum up
    with pytest.raises(ValueError, match=r"loss_combined requires at least one loss function to sum up"):
        loss_combined(x, [])

    # Missing 'loss'
    with pytest.raises(IndexError, match=r"tuple index out of range"):
        loss_combined(x, [()])


def test_quadratic_cost_approximation_l2() -> None:
    """Test quadratic_cost_approximation_l2 on single timesteps."""
    from donk.costs import QuadraticCosts

    cost_function = QuadraticCosts.quadratic_cost_approximation_l2(
        t=np.array([1, 2, 3]),
        w=np.array([1, 1, 2]),
    )

    assert_allclose(cost_function.C, np.diag([1, 1, 2]))
    assert_allclose(cost_function.c, [-1, -2, -6])
    assert_allclose(cost_function.cc, 0.5 + 2 + 9)


def test_quadratic_cost_approximation_l2_batched() -> None:
    """Test quadratic_cost_approximation_l2 on a trajectory."""
    from donk.costs import QuadraticCosts

    cost_function = QuadraticCosts.quadratic_cost_approximation_l2(
        t=np.array([[-2, 2, 0], [-1, 1, 0], [0, 0, 0]]),
        w=np.array([[1, 1, 2], [1, 1, 2], [10, 10, 0]]),
    )

    assert_allclose(
        cost_function.C,
        [np.diag([1, 1, 2]), np.diag([1, 1, 2]), np.diag([10, 10, 0])],
    )
    assert_allclose(
        cost_function.c,
        [[2, -2, 0], [1, -1, 0], [0, 0, 0]],
    )
    assert_allclose(cost_function.cc, [4, 1, 0])


def test_quadratic_cost_approximation_l1() -> None:
    """Test quadratic_cost_approximation_l1 on single timesteps."""
    from donk.costs import QuadraticCosts

    cost_function = QuadraticCosts.quadratic_cost_approximation_l1(
        xu=np.array([3, 2]),
        t=np.array([2, 2]),
        w=np.array([1, 2]),
        alpha=1e-2,
    )

    assert_allclose(cost_function.C, np.diag([0.01 / 1.01**1.5, 20]))
    assert_allclose(cost_function.c, [1 / 1.01**0.5 - 0.03 / 1.01**1.5, -40])
    assert_allclose(cost_function.cc, 1.01**0.5 - 3 / 1.01**0.5 + 0.045 / 1.01**1.5 + 0.2 + 40)


def test_quadratic_cost_approximation_l1_batched() -> None:
    """Test quadratic_cost_approximation_l1 on a trajectory."""
    from donk.costs import QuadraticCosts

    cost_function = QuadraticCosts.quadratic_cost_approximation_l1(
        xu=np.array([[-2, 0], [0, 0]]),
        t=np.array([[1, 2], [10, 0]]),
        w=np.array([[1, 2], [10, 0]]),
        alpha=1e-6,
    )

    assert_allclose(cost_function.C, np.zeros((2, 2, 2)), atol=1e-6)
    assert_allclose(cost_function.c, [[-1, -2], [-10, 0]], atol=1e-6)
    assert_allclose(cost_function.cc, [3 * 1 - 2 + 2 * 2, 10 * 10], atol=1e-6)


def test_quadratic_costs_compute_costs() -> None:
    from donk.costs import QuadraticCosts

    rng = np.random.default_rng(0)
    T, dX, dXU = 10, 3, 5

    target = rng.standard_normal((T, dXU))
    target[-1, dX:] = 0  # No action at final state
    weights = rng.standard_normal((T, dXU))

    cost_function = QuadraticCosts.quadratic_cost_approximation_l2(target, weights)

    # When targets reached, costs are zero
    assert_allclose(cost_function.compute_costs(target[:, :dX], target[:-1, dX:]), np.zeros(T))

    XU = rng.standard_normal((T, dXU))
    XU[-1, dX:] = 0  # No action at final state
    assert_allclose(
        cost_function.compute_costs(XU[:, :dX], XU[:-1, dX:]),
        np.sum(weights * (target - XU) ** 2, axis=-1) / 2,
    )


def test_quadratic_costs_expected_costs() -> None:
    from donk.costs import QuadraticCosts
    from donk.samples import TrajectoryDistribution
    from donk.testing import random_spd

    rng = np.random.default_rng(0)
    N, T, dX, dXU = 1000, 10, 3, 5

    target = rng.standard_normal((T, dXU))
    weights = rng.standard_normal((T, dXU))
    cost_function = QuadraticCosts.quadratic_cost_approximation_l2(target, weights)

    traj = TrajectoryDistribution(mean=rng.standard_normal((T, dXU)), covar=random_spd((T, dXU, dXU), rng), dX=3)

    XU = np.empty((N, T, dXU))
    for t in range(T):
        XU[:, t] = rng.multivariate_normal(traj.mean[t], traj.covar[t], N)

    mean_costs = np.mean(cost_function.compute_costs(XU[:, :, :dX], XU[:, :-1, dX:]), axis=0)

    assert_allclose(cost_function.expected_costs(traj)[:-1], mean_costs[:-1], rtol=0.25)


def test_quadratic_costs_add() -> None:
    from donk.costs import QuadraticCosts

    rng = np.random.default_rng(0)
    T, dXU = 10, 5

    cost_function_1 = QuadraticCosts.quadratic_cost_approximation_l2(
        t=rng.standard_normal((T, dXU)),
        w=rng.standard_normal((T, dXU)),
    )
    cost_function_2 = QuadraticCosts.quadratic_cost_approximation_l1(
        xu=rng.standard_normal((T, dXU)),
        t=rng.standard_normal((T, dXU)),
        w=rng.standard_normal((T, dXU)),
        alpha=1e-6,
    )

    cost_function = cost_function_1 + cost_function_2

    assert_allclose(cost_function.C, cost_function_1.C + cost_function_2.C)
    assert_allclose(cost_function.c, cost_function_1.c + cost_function_2.c)
    assert_allclose(cost_function.cc, cost_function_1.cc + cost_function_2.cc)

    with pytest.raises(TypeError, match=r"unsupported operand type"):
        cost_function + 5


def test_quadratic_costs_mul() -> None:
    from donk.costs import QuadraticCosts

    rng = np.random.default_rng(0)
    T, dXU = 10, 5

    cost_function_1 = QuadraticCosts.quadratic_cost_approximation_l2(
        t=rng.standard_normal((T, dXU)),
        w=rng.standard_normal((T, dXU)),
    )

    cost_function = cost_function_1 * 2

    assert_allclose(cost_function.C, cost_function_1.C * 2)
    assert_allclose(cost_function.c, cost_function_1.c * 2)
    assert_allclose(cost_function.cc, cost_function_1.cc * 2)

    with pytest.raises(TypeError, match=r"unsupported operand type"):
        cost_function * cost_function_1


def test_quadratic_costs_rmul() -> None:
    from donk.costs import QuadraticCosts

    rng = np.random.default_rng(0)
    T, dXU = 10, 5

    cost_function_1 = QuadraticCosts.quadratic_cost_approximation_l2(
        t=rng.standard_normal((T, dXU)),
        w=rng.standard_normal((T, dXU)),
    )

    cost_function = 1.2 * cost_function_1

    assert_allclose(cost_function.C, cost_function_1.C * 1.2)
    assert_allclose(cost_function.c, cost_function_1.c * 1.2)
    assert_allclose(cost_function.cc, cost_function_1.cc * 1.2)


def test_symbolic_cost_function() -> None:
    from donk.costs import SymbolicCostFunction

    T, dX, dU = 2, 1, 1

    def cost_fun(X: np.ndarray, U: np.ndarray) -> np.ndarray:
        c = np.sum(X**2, axis=-1) / 2
        c[:-1] += np.sum(U**2, axis=-1) / 2
        return c

    cost_function = SymbolicCostFunction(cost_fun, T, dX, dU)

    X = np.array([[0], [1], [2]])
    U = np.array([[1], [-1]])

    costs = cost_function.quadratic_approximation(X, U)

    assert_allclose(costs.C, [np.diag([1, 1]), np.diag([1, 1]), np.diag([1, 0])], atol=1e-16)
    assert_allclose(costs.c, [[0, 0], [0, 0], [0, 0]], atol=1e-16)
    assert_allclose(costs.cc, [0, 0, 0], atol=1e-16)


def test_symbolic_cost_function_2() -> None:
    from donk.costs import QuadraticCosts, SymbolicCostFunction

    T, dX, dU = 2, 1, 1

    def cost_fun(X: np.ndarray, U: np.ndarray) -> np.ndarray:
        return [
            (np.sum((X[0] - 1) ** 2) + 0.5 * np.sum((U[0] - 2) ** 2)) / 2,
            (np.sum(2 * (X[1] + 1) ** 2) - np.sum((U[1] + 1) ** 2)) / 2,
            (np.sum(X[2] ** 2)) / 2,
        ]

    cost_function = SymbolicCostFunction(cost_fun, T, dX, dU)

    X = np.array([[0], [1], [2]])
    U = np.array([[1], [-1]])

    costs = cost_function.quadratic_approximation(X, U)
    costs_tgt = QuadraticCosts.quadratic_cost_approximation_l2(
        t=np.array([[1, 2], [-1, -1], [0, 0]]),
        w=np.array([[1, 0.5], [2, -1], [1, 0]]),
    )

    assert_allclose(costs.C, costs_tgt.C, atol=1e-16)
    assert_allclose(costs.c, costs_tgt.c, atol=1e-16)
    assert_allclose(costs.cc, costs_tgt.cc, atol=1e-16)


def test_symbolic_cost_function_compute_costs() -> None:
    from donk.costs import SymbolicCostFunction

    T, dX, dU = 2, 1, 1

    def cost_fun(X: np.ndarray, U: np.ndarray) -> np.ndarray:
        return np.array([
            (np.sum((X[0] - 1) ** 2) + 0.5 * np.sum((U[0] - 2) ** 2)) / 2,
            (np.sum(2 * (X[1] + 1) ** 2) - np.sum((U[1] + 1) ** 2)) / 2,
            (np.sum(X[2] ** 2)) / 2,
        ])

    cost_function = SymbolicCostFunction(cost_fun, T, dX, dU)

    X = np.array([[0], [1], [2]])  # (T+1, dX)
    U = np.array([[1], [-1]])  # (T, dU)

    # Flat input
    assert_allclose(cost_function.compute_costs(X, U), [0.75, 4, 2])

    # Nested input
    assert_allclose(cost_function.compute_costs([X, X], [U, U]), [[0.75, 4, 2], [0.75, 4, 2]])


def test_symbolic_cost_function_compute_costs_is_numeric() -> None:
    from donk.costs import SymbolicCostFunction

    T, dX, dU = 2, 1, 1

    def cost_fun(X: np.ndarray, U: np.ndarray) -> np.ndarray:
        import sympy

        return np.array([
            sympy.sin(X[0, 0]) + sympy.sqrt(U[0, 0]),
            sympy.log(X[1, 0]),
            X[2, 0] ** 2 / 2,
        ])

    cost_function = SymbolicCostFunction(cost_fun, T, dX, dU)

    X = np.array([[0], [1], [2]])  # (T+1, dX)
    U = np.array([[1], [-1]])  # (T, dU)

    # No sympy types present
    costs = cost_function.compute_costs(X, U)
    assert costs.dtype == np.float64
    assert_allclose(costs, [1, 0, 2])


def test_multipart_symbolic_cost_function_compute_costs() -> None:
    from donk.costs import MultipartSymbolicCostFunction

    T, dX, dU = 2, 1, 1

    def state_costs(X: np.ndarray, U: np.ndarray) -> np.ndarray:
        return np.array([np.sum((X[0] - 1) ** 2) / 2, np.sum(2 * (X[1] + 1) ** 2) / 2, np.sum(X[2] ** 2) / 2])

    def action_costs(X: np.ndarray, U: np.ndarray) -> np.ndarray:
        return np.array([0.5 * np.sum((U[0] - 2) ** 2) / 2, -np.sum((U[1] + 1) ** 2) / 2, 0])

    cost_function = MultipartSymbolicCostFunction([state_costs, action_costs], ["state", "action"], T, dX, dU)

    X = np.array([[0], [1], [2]])  # (T+1, dX)
    U = np.array([[1], [-1]])  # (T, dU)

    # Flat input
    assert_allclose(cost_function.compute_costs(X, U), [0.75, 4, 2])

    # Nested input
    assert_allclose(cost_function.compute_costs([X, X], [U, U]), [[0.75, 4, 2], [0.75, 4, 2]])

    # Individual costs
    part_costs = cost_function.compute_costs_individual(X, U)

    assert_allclose(part_costs["state"], [0.5, 4, 2])

    assert_allclose(part_costs["action"], [0.25, 0, 0])

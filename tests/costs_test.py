import unittest

import numpy as np
from numpy.testing import assert_allclose

from tests.utils import random_spd


def _create_symbols(D):
    from sympy import symbols

    d_sym = symbols(f'd0:{D}')
    wp_sym = symbols(f'w0:{D}')

    return d_sym, wp_sym


def _evaluate_loss(d, wp, d_sym, wp_sym, loss):
    from sympy import Matrix, diff

    T, D = d.shape

    # Derivatives
    d1 = Matrix([diff(loss, s) for s in d_sym])
    d2 = Matrix([[diff(d, s) for s in d_sym] for d in d1])

    l = np.empty((T, ))
    lx = np.empty((T, D))
    lxx = np.empty((T, D, D))
    for t in range(T):  # Evaluate by substituting actual values
        substitutions = {d_sym[i]: d[t, i] for i in range(D)}
        substitutions.update({wp_sym[i]: wp[t, i] for i in range(D)})

        l[t] = float(loss.subs(substitutions))
        lx[t] = np.array(d1.subs(substitutions), dtype=float).reshape(D)
        lxx[t] = np.array(d2.subs(substitutions), dtype=float).reshape(D, D)
    return l, lx, lxx


def loss_l2_ref(x, t, w):
    """Reference implmentation for loss_l2 using sympy for symbolic differentation."""
    _, dX = x.shape
    d_sym, wp_sym = _create_symbols(dX)

    # Loss function
    loss = 0.5 * sum(d_sym[i]**2 * wp_sym[i] for i in range(dX))

    return _evaluate_loss(x - t, w, d_sym, wp_sym, loss)


def loss_l1_ref(x, t, w, alpha):
    """Reference implmentation for loss_l1 using sympy for symbolic differentation."""
    from sympy import sqrt

    _, dX = x.shape
    d_sym, wp_sym = _create_symbols(dX)

    # Loss function
    loss = sum(sqrt(d_sym[i]**2 + alpha) * wp_sym[i] for i in range(dX))

    return _evaluate_loss(x - t, w, d_sym, wp_sym, loss)


def loss_log_cosh_ref(x, t, w):
    """Reference implmentation for loss_l2 using sympy for symbolic differentation."""
    from sympy import cosh, log
    _, dX = x.shape
    d_sym, wp_sym = _create_symbols(dX)

    # Loss function
    loss = sum([log(cosh(d_sym[i])) * wp_sym[i] for i in range(dX)])

    return _evaluate_loss(x - t, w, d_sym, wp_sym, loss)


class Test_Losses(unittest.TestCase):

    def test_loss_l2(self):
        """Test loss_l2 implementation agains reference implementation using random values."""
        from donk.costs import loss_l2

        T, dX = 10, 3
        x = np.random.randn(T, dX)
        t = np.random.randn(T, dX)
        w = np.random.uniform(size=(T, dX))

        l_ref, lx_ref, lxx_ref = loss_l2_ref(x, t, w)
        l, lx, lxx = loss_l2(x, t, w)

        assert_allclose(l_ref, l)
        assert_allclose(lx_ref, lx)
        assert_allclose(lxx_ref, lxx)

    def test_loss_l1(self):
        """Test loss_l1 implementation agains reference implementation using random values."""
        from donk.costs import loss_l1

        T, dX = 10, 3
        x = np.random.randn(T, dX)
        t = np.random.randn(T, dX)
        w = np.random.uniform(size=(T, dX))

        l_ref, lx_ref, lxx_ref = loss_l1_ref(x, t, w, alpha=1e-2)
        l, lx, lxx = loss_l1(x, t, w, alpha=1e-2)

        assert_allclose(l_ref, l)
        assert_allclose(lx_ref, lx)
        assert_allclose(lxx_ref, lxx)

    def test_loss_log_cosh(self):
        """Test loss_log_cosh implementation agains reference implementation using random values."""
        from donk.costs import loss_log_cosh

        T, dX = 10, 3
        x = np.random.randn(T, dX)
        t = np.random.randn(T, dX)
        w = np.random.uniform(size=(T, dX))

        l_ref, lx_ref, lxx_ref = loss_log_cosh_ref(x, t, w)
        l, lx, lxx = loss_log_cosh(x, t, w)

        assert_allclose(l_ref, l)
        assert_allclose(lx_ref, lx)
        assert_allclose(lxx_ref, lxx)

    def test_loss_sum(self):
        """Test loss_combined summing up two losses."""
        from donk.costs import loss_combined, loss_l2, loss_log_cosh

        T, dX = 10, 3
        x = np.random.randn(T, dX)
        t = np.random.randn(T, dX)
        w = np.random.uniform(size=(T, dX))

        # Reference
        l_l2_ref, lx_l2_ref, lxx_l2_ref = loss_l2(x, t, w)
        l_log_cosh_ref, lx_log_cosh_ref, lxx_log_cosh_ref = loss_log_cosh_ref(x, t, w)
        l_ref = l_l2_ref + l_log_cosh_ref
        lx_ref = lx_l2_ref + lx_log_cosh_ref
        lxx_ref = lxx_l2_ref + lxx_log_cosh_ref

        # Impl
        l, lx, lxx = loss_combined(x, [
            (loss_l2, {
                't': t,
                'w': w,
            }),
            (loss_log_cosh, {
                't': t,
                'w': w,
            }),
        ])

        assert_allclose(l_ref, l)
        assert_allclose(lx_ref, lx)
        assert_allclose(lxx_ref, lxx)

    def test_loss_sum_single_argument(self):
        """Test loss_combined with only a single loss to sum up."""
        from donk.costs import loss_combined, loss_l2

        T, dX = 10, 3
        x = np.random.randn(T, dX)
        t = np.random.randn(T, dX)
        w = np.random.uniform(size=(T, dX))

        # Reference
        l_ref, lx_ref, lxx_ref = loss_l2_ref(x, t, w)

        # Impl
        l, lx, lxx = loss_combined(x, [
            (loss_l2, {
                't': t,
                'w': w,
            }),
        ])

        assert_allclose(l_ref, l)
        assert_allclose(lx_ref, lx)
        assert_allclose(lxx_ref, lxx)

    def test_loss_sum_missing_arguments(self):
        """Test loss_combined raises errors on missing arguments."""
        from donk.costs import loss_combined

        T, dX = 10, 3
        x = np.random.randn(T, dX)

        # No losses to sum up
        with self.assertRaises(ValueError):
            loss_combined(x, [])

        # Missing 'loss'
        with self.assertRaises(IndexError):
            loss_combined(x, [
                (),
            ])


class Test_QuadraticCosts(unittest.TestCase):

    def test_quadratic_cost_approximation_l2(self):
        """Test quadratic_cost_approximation_l2 on single timesteps."""
        from donk.costs import QuadraticCosts

        cost_function = QuadraticCosts.quadratic_cost_approximation_l2(
            t=np.array([1, 2, 3]),
            w=np.array([1, 1, 2]),
        )

        assert_allclose(cost_function.C, np.diag([1, 1, 2]))
        assert_allclose(cost_function.c, [-1, -2, -6])
        assert_allclose(cost_function.cc, 0.5 + 2 + 9)

    def test_quadratic_cost_approximation_l2_batched(self):
        """Test quadratic_cost_approximation_l2 on a trajectory."""
        from donk.costs import QuadraticCosts

        cost_function = QuadraticCosts.quadratic_cost_approximation_l2(
            t=np.array([
                [-2, 2, 0],
                [-1, 1, 0],
                [0, 0, 0],
            ]),
            w=np.array([
                [1, 1, 2],
                [1, 1, 2],
                [10, 10, 0],
            ]),
        )

        assert_allclose(cost_function.C, [
            np.diag([1, 1, 2]),
            np.diag([1, 1, 2]),
            np.diag([10, 10, 0]),
        ])
        assert_allclose(cost_function.c, [
            [2, -2, 0],
            [1, -1, 0],
            [0, 0, 0],
        ])
        assert_allclose(cost_function.cc, [4, 1, 0])

    def test_quadratic_cost_approximation_l1(self):
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

    def test_quadratic_cost_approximation_l1_batched(self):
        """Test quadratic_cost_approximation_l1 on a trajectory."""
        from donk.costs import QuadraticCosts

        cost_function = QuadraticCosts.quadratic_cost_approximation_l1(
            xu=np.array([
                [-2, 0],
                [0, 0],
            ]),
            t=np.array([
                [1, 2],
                [10, 0],
            ]),
            w=np.array([
                [1, 2],
                [10, 0],
            ]),
            alpha=1e-6,
        )

        assert_allclose(cost_function.C, np.zeros((2, 2, 2)), atol=1e-6)
        assert_allclose(cost_function.c, [
            [-1, -2],
            [-10, 0],
        ], atol=1e-6)
        assert_allclose(cost_function.cc, [3 * 1 - 2 + 2 * 2, 10 * 10], atol=1e-6)

    def test_compute_costs(self):
        """Test QuadraticCosts.compute_costs."""
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
            np.sum(weights * (target - XU)**2, axis=-1) / 2,
        )

    def test_expected_costs(self):
        """Test QuadraticCosts.expected_costs."""
        from donk.costs import QuadraticCosts
        from donk.samples import TrajectoryDistribution

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

        assert_allclose(cost_function.expected_costs(traj)[:-1], mean_costs[:-1], rtol=.25)

    def test_add(self):
        """Test QuadraticCosts.expected_costs."""
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

        with self.assertRaises(TypeError):
            cost_function + 5

    def test_mul(self):
        """Test QuadraticCosts.expected_costs."""
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

        with self.assertRaises(TypeError):
            cost_function * cost_function_1

    def test_rmul(self):
        """Test QuadraticCosts.expected_costs."""
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


class Test_CostFunction(unittest.TestCase):

    def test_symbolic_cost_function(self):
        from donk.costs import SymbolicCostFunction

        T, dX, dU = 2, 1, 1

        def cost_fun(X, U):
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

    def test_symbolic_cost_function_2(self):
        from donk.costs import QuadraticCosts, SymbolicCostFunction

        T, dX, dU = 2, 1, 1

        def cost_fun(X, U):
            return [
                (np.sum((X[0] - 1)**2) + 0.5 * np.sum((U[0] - 2)**2)) / 2,
                (np.sum(2 * (X[1] + 1)**2) - np.sum((U[1] + 1)**2)) / 2,
                (np.sum(X[2]**2)) / 2,
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

    def test_compute_costs(self):
        from donk.costs import SymbolicCostFunction

        T, dX, dU = 2, 1, 1

        def cost_fun(X, U):
            return np.array(
                [
                    (np.sum((X[0] - 1)**2) + 0.5 * np.sum((U[0] - 2)**2)) / 2,
                    (np.sum(2 * (X[1] + 1)**2) - np.sum((U[1] + 1)**2)) / 2,
                    (np.sum(X[2]**2)) / 2,
                ]
            )

        cost_function = SymbolicCostFunction(cost_fun, T, dX, dU)

        X = np.array([[0], [1], [2]])  # (T+1, dX)
        U = np.array([[1], [-1]])  # (T, dU)

        # Flat input
        assert_allclose(cost_function.compute_costs(X, U), [.75, 4, 2])

        # Nested input
        assert_allclose(cost_function.compute_costs([X, X], [U, U]), [[.75, 4, 2], [.75, 4, 2]])

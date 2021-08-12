import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal

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

        assert_array_almost_equal(l_ref, l)
        assert_array_almost_equal(lx_ref, lx)
        assert_array_almost_equal(lxx_ref, lxx)

    def test_loss_l1(self):
        """Test loss_l1 implementation agains reference implementation using random values."""
        from donk.costs import loss_l1

        T, dX = 10, 3
        x = np.random.randn(T, dX)
        t = np.random.randn(T, dX)
        w = np.random.uniform(size=(T, dX))

        l_ref, lx_ref, lxx_ref = loss_l1_ref(x, t, w, alpha=1e-2)
        l, lx, lxx = loss_l1(x, t, w, alpha=1e-2)

        assert_array_almost_equal(l_ref, l)
        assert_array_almost_equal(lx_ref, lx)
        assert_array_almost_equal(lxx_ref, lxx)

    def test_loss_log_cosh(self):
        """Test loss_log_cosh implementation agains reference implementation using random values."""
        from donk.costs import loss_log_cosh

        T, dX = 10, 3
        x = np.random.randn(T, dX)
        t = np.random.randn(T, dX)
        w = np.random.uniform(size=(T, dX))

        l_ref, lx_ref, lxx_ref = loss_log_cosh_ref(x, t, w)
        l, lx, lxx = loss_log_cosh(x, t, w)

        assert_array_almost_equal(l_ref, l)
        assert_array_almost_equal(lx_ref, lx)
        assert_array_almost_equal(lxx_ref, lxx)

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

        assert_array_almost_equal(l_ref, l)
        assert_array_almost_equal(lx_ref, lx)
        assert_array_almost_equal(lxx_ref, lxx)

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

        assert_array_almost_equal(l_ref, l)
        assert_array_almost_equal(lx_ref, lx)
        assert_array_almost_equal(lxx_ref, lxx)

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

        assert_array_almost_equal(cost_function.C, np.diag([1, 1, 2]))
        assert_array_almost_equal(cost_function.c, [-1, -2, -6])
        assert_array_almost_equal(cost_function.cc, 0.5 + 2 + 9)

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

        assert_array_almost_equal(cost_function.C, [
            np.diag([1, 1, 2]),
            np.diag([1, 1, 2]),
            np.diag([10, 10, 0]),
        ])
        assert_array_almost_equal(cost_function.c, [
            [2, -2, 0],
            [1, -1, 0],
            [0, 0, 0],
        ])
        assert_array_almost_equal(cost_function.cc, [4, 1, 0])

    def test_quadratic_cost_approximation_l1(self):
        """Test quadratic_cost_approximation_l1 on single timesteps."""
        from donk.costs import QuadraticCosts

        cost_function = QuadraticCosts.quadratic_cost_approximation_l1(
            xu=np.array([3, 2]),
            t=np.array([2, 2]),
            w=np.array([1, 2]),
            alpha=1e-2,
        )

        assert_array_almost_equal(cost_function.C, np.diag([0.01 / 1.01**1.5, 20]))
        assert_array_almost_equal(cost_function.c, [1 / 1.01**0.5 - 0.03 / 1.01**1.5, -40])
        assert_array_almost_equal(cost_function.cc, 1.01**0.5 - 3 / 1.01**0.5 + 0.045 / 1.01**1.5 + 0.2 + 40)

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

        assert_array_almost_equal(cost_function.C, np.zeros((2, 2, 2)))
        assert_array_almost_equal(cost_function.c, [
            [-1, -2],
            [-10, 0],
        ])
        assert_array_almost_equal(cost_function.cc, [3 * 1 - 2 + 2 * 2, 10 * 10])

    def test_compute_costs(self):
        """Test QuadraticCosts.compute_costs."""
        from donk.costs import QuadraticCosts

        rng = np.random.default_rng(0)
        T, dXU = 10, 5

        target = rng.standard_normal((T, dXU))
        weights = rng.standard_normal((T, dXU))

        cost_function = QuadraticCosts.quadratic_cost_approximation_l2(target, weights)

        # When targets reached, costs are zero
        assert_array_almost_equal(cost_function.compute_costs(target), np.zeros(T))

        X = rng.standard_normal((T, dXU))
        assert_array_almost_equal(cost_function.compute_costs(X), np.sum(weights * (target - X)**2, axis=-1) / 2)

    def test_expected_costs(self):
        """Test QuadraticCosts.expected_costs."""
        from donk.costs import QuadraticCosts

        rng = np.random.default_rng(0)
        N, T, dXU = 1000, 10, 5

        target = rng.standard_normal((T, dXU))
        weights = rng.standard_normal((T, dXU))
        cost_function = QuadraticCosts.quadratic_cost_approximation_l2(target, weights)

        traj_mean = rng.standard_normal((T, dXU))
        traj_covar = random_spd((T, dXU, dXU), rng)

        XU = np.empty((N, T, dXU))
        for t in range(T):
            XU[:, t] = rng.multivariate_normal(traj_mean[t], traj_covar[t], N)

        mean_costs = np.mean(cost_function.compute_costs(XU), axis=0)

        assert_allclose(cost_function.expected_costs(traj_mean, traj_covar), mean_costs, rtol=.25)

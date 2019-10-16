import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal


def _create_symbols(D):
    from sympy import MatrixSymbol

    d_sym = np.array(MatrixSymbol('d', 1, D))[0]
    wp_sym = np.array(MatrixSymbol('wp', 1, D))[0]

    return d_sym, wp_sym


def _evaluate_loss(d, wp, d_sym, wp_sym, loss):
    from sympy import diff

    T, D = d.shape

    # Derivatives
    d1 = diff(loss, d_sym)
    d2 = diff(d1, d_sym)

    l = np.empty((T, ))
    lx = np.empty((T, D))
    lxx = np.empty((T, D, D))
    for t in range(T):  # Evaluate by substituting actual values
        substitutions = {d_sym[i]: d[t, i] for i in range(D)}
        substitutions.update({wp_sym[i]: wp[t, i] for i in range(D)})

        l[t] = float(loss.subs(substitutions))
        lx[t] = np.array(d1.subs(substitutions), dtype=float)
        lxx[t] = np.array(d2.subs(substitutions), dtype=float).reshape(D, D)
    return l, lx, lxx


def loss_l2_ref(d, wp):
    """Reference implmentation for loss_l2 using sympy for symbolic differentation."""
    _, D = d.shape
    d_sym, wp_sym = _create_symbols(D)

    # Loss function
    loss = 0.5 * sum(d_sym**2 * wp_sym)

    return _evaluate_loss(d, wp, d_sym, wp_sym, loss)


def loss_log_cosh_ref(d, wp):
    """Reference implmentation for loss_l2 using sympy for symbolic differentation."""
    from sympy import log, cosh
    _, D = d.shape
    d_sym, wp_sym = _create_symbols(D)

    # Loss function
    loss = sum([log(cosh(d_sym[i])) * wp_sym[i] for i in range(D)])

    return _evaluate_loss(d, wp, d_sym, wp_sym, loss)


class Test_Losses(unittest.TestCase):
    def test_loss_l2(self):
        """Test loss_l2 implementation agains reference implementation using random values."""
        from donk.costs import loss_l2

        T, D = 10, 3
        d = np.random.randn(T, D)
        wp = np.random.uniform(size=(T, D))

        l_ref, lx_ref, lxx_ref = loss_l2_ref(d, wp)
        l, lx, lxx = loss_l2(d, wp)

        assert_array_almost_equal(l_ref, l)
        assert_array_almost_equal(lx_ref, lx)
        assert_array_almost_equal(lxx_ref, lxx)

    def test_loss_log_cosh(self):
        """Test loss_log_cosh implementation agains reference implementation using random values."""
        from donk.costs import loss_log_cosh

        T, D = 10, 3
        d = np.random.randn(T, D)
        wp = np.random.uniform(size=(T, D))

        l_ref, lx_ref, lxx_ref = loss_log_cosh_ref(d, wp)
        l, lx, lxx = loss_log_cosh(d, wp)

        assert_array_almost_equal(l_ref, l)
        assert_array_almost_equal(lx_ref, lx)
        assert_array_almost_equal(lxx_ref, lxx)

    def test_loss_sum(self):
        """Test loss_combined summing up two losses."""
        from donk.costs import loss_log_cosh, loss_l2, loss_combined

        T, D = 10, 3
        d = np.random.randn(T, D)
        wp = np.random.uniform(size=(T, D))

        # Reference
        l_l2_ref, lx_l2_ref, lxx_l2_ref = loss_l2(d, wp)
        l_log_cosh_ref, lx_log_cosh_ref, lxx_log_cosh_ref = loss_log_cosh_ref(d, wp)
        l_ref = l_l2_ref + l_log_cosh_ref
        lx_ref = lx_l2_ref + lx_log_cosh_ref
        lxx_ref = lxx_l2_ref + lxx_log_cosh_ref

        # Impl
        l, lx, lxx = loss_combined(d, wp, [
            {
                'loss': loss_l2,
                'kwargs': {}
            },
            {
                'loss': loss_log_cosh,
            },
        ])

        assert_array_almost_equal(l_ref, l)
        assert_array_almost_equal(lx_ref, lx)
        assert_array_almost_equal(lxx_ref, lxx)

    def test_loss_sum_single_argument(self):
        """Test loss_combined with only a single loss to sum up."""
        from donk.costs import loss_l2, loss_combined

        T, D = 10, 3
        d = np.random.randn(T, D)
        wp = np.random.uniform(size=(T, D))

        # Reference
        l_ref, lx_ref, lxx_ref = loss_l2_ref(d, wp)

        # Impl
        l, lx, lxx = loss_combined(d, wp, [{'loss': loss_l2}])

        assert_array_almost_equal(l_ref, l)
        assert_array_almost_equal(lx_ref, lx)
        assert_array_almost_equal(lxx_ref, lxx)

    def test_loss_sum_missing_arguments(self):
        """Test loss_combined raises errors on missing arguments."""
        from donk.costs import loss_combined

        T, D = 10, 3
        d = np.random.randn(T, D)
        wp = np.random.uniform(size=(T, D))

        # No losses to sum up
        with self.assertRaises(ValueError):
            loss_combined(d, wp, [])

        # Missing 'loss'
        with self.assertRaises(KeyError):
            loss_combined(d, wp, [
                {},
            ])

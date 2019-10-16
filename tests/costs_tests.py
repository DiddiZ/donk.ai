import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal


def loss_l2_ref(d, wp):
    """Reference implmentation for loss_l2 using sympy for symbolic differentation."""
    from sympy import MatrixSymbol, diff
    T, D = d.shape

    d_sym = np.array(MatrixSymbol('d', 1, D))[0]
    wp_sym = np.array(MatrixSymbol('wp', 1, D))[0]

    # Loss function and derivatives
    loss = 0.5 * sum(d_sym**2 * wp_sym)
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

import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from tests.utils import random_spd


class Test_Batches(unittest.TestCase):

    def test_batched_inv_spd(self):
        """Test batched_inv_spd implementation agains reference implementation using random values."""
        from donk.utils.batched import batched_inv_spd, batched_cholesky

        T, dX = 10, 10

        a = np.empty((T, dX, dX))
        a_chol_ref = np.empty_like(a)
        a_inv_ref = np.empty_like(a)
        for t in range(T):
            a[t] = random_spd(dX)
            a_chol_ref[t] = np.linalg.cholesky(a[t])
            a_inv_ref[t] = np.linalg.inv(a[t])

        a_chol = batched_cholesky(a)
        a_inv = batched_inv_spd(a_chol)

        assert_array_almost_equal(a_chol_ref, a_chol)
        assert_array_almost_equal(a_inv_ref, a_inv)

    def test_symmetrize(self):
        from donk.utils import symmetrize

        rng = np.random.default_rng(0)
        A = rng.normal(size=(5, 5))
        A_sym = symmetrize(A)

        self.assertIs(A_sym, A)
        assert_array_equal(A_sym, A_sym.T)

    def test_symmetrize_batched(self):
        from donk.utils import symmetrize

        rng = np.random.default_rng(0)
        A = rng.normal(size=(3, 5, 5))
        A_sym = symmetrize(A)

        self.assertIs(A_sym, A)
        self.assertTupleEqual(A_sym.shape, (3, 5, 5))
        assert_array_equal(A_sym[0], A_sym[0].T)
        assert_array_equal(A_sym[1], A_sym[1].T)
        assert_array_equal(A_sym[2], A_sym[2].T)

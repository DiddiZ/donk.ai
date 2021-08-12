import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from tests.utils import random_spd


class Test_TransitionsPool(unittest.TestCase):

    def test_add(self):
        """Test TransitionsPool.add()."""
        from donk.samples import TransitionsPool

        N, T, dX, dU = 3, 10, 5, 3
        rng = np.random.default_rng(0)

        X = rng.standard_normal((N, T + 1, dX))
        U = rng.standard_normal((N, T, dU))
        pool = TransitionsPool()
        pool.add(X, U)

        # Retrieve all
        assert_array_equal(pool.get_transitions(), np.c_[X[:, :-1].reshape(-1, dX), U.reshape(-1, dU), X[:, 1:].reshape(-1, dX)])

        # Retrieve N
        assert_array_equal(
            pool.get_transitions(N=5), np.c_[X[2, -6:-1].reshape(-1, dX), U[2, -5:].reshape(-1, dU), X[2, -5:].reshape(-1, dX)]
        )

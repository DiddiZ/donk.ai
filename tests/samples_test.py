import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from tests.utils import random_spd


class Test_TransitionPool(unittest.TestCase):
    def test_add(self):
        """Test TransitionPool.add()."""
        from donk.samples import TransitionPool

        N, T, dX, dU = 3, 10, 5, 3
        rng = np.random.default_rng(0)

        X = rng.standard_normal((N, T + 1, dX))
        U = rng.standard_normal((N, T, dU))
        pool = TransitionPool()
        pool.add(X, U)

        # Retrieve all
        assert_array_equal(pool.get_transitions(), np.c_[X[:, :-1].reshape(-1, dX), U.reshape(-1, dU), X[:, 1:].reshape(-1, dX)])

        # Retrieve N
        assert_array_equal(
            pool.get_transitions(N=5), np.c_[X[2, -6:-1].reshape(-1, dX), U[2, -5:].reshape(-1, dU), X[2, -5:].reshape(-1, dX)]
        )


class Test_TrajectoryDistribution(unittest.TestCase):
    def test_init(self):
        """Test TrajectoryDistribution.sample()."""
        from donk.samples import TrajectoryDistribution

        T, dX, dU = 10, 5, 3
        rng = np.random.default_rng(0)

        mean = rng.standard_normal((T, dX + dU))
        covar = random_spd((T, dX + dU, dX + dU), rng)
        traj = TrajectoryDistribution(mean, covar, dX)

        assert_array_equal(traj.mean, mean)
        assert_array_equal(traj.covar, covar)

        assert_array_equal(traj.X_mean, mean[:, :dX])
        assert_array_equal(traj.U_mean, mean[:-1, dX:])

        assert_array_equal(traj.X_covar, covar[:, :dX, :dX])
        assert_array_equal(traj.U_covar, covar[:-1, dX:, dX:])

    def test_sample(self):
        """Test TrajectoryDistribution.sample()."""
        from donk.samples import TrajectoryDistribution

        T, dX, dU = 10, 5, 3
        rng = np.random.default_rng(1)

        mean = rng.standard_normal((T + 1, dX + dU))
        covar = random_spd((T + 1, dX + dU, dX + dU), rng)
        traj = TrajectoryDistribution(mean, covar, dX)

        X, U = traj.sample((13, 14), rng)
        self.assertTupleEqual(X.shape, (13, 14, T + 1, dX))
        self.assertTupleEqual(U.shape, (13, 14, T, dU))

        assert_allclose(np.mean(np.mean(X, axis=0), axis=0), mean[:, :dX], atol=1)
        assert_allclose(np.mean(np.mean(U, axis=0), axis=0), mean[:-1, dX:], atol=1)

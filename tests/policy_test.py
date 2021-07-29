import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal
from tests.utils import random_spd


class Test_LinearGaussianPolicy(unittest.TestCase):

    def test_init_from_pol_covar(self):
        """Test __init__ using pol_covar."""
        from donk.policy import LinearGaussianPolicy

        T, dX, dU = 10, 10, 6
        rng = np.random.default_rng(0)

        K = rng.normal(size=(T, dU, dX))
        k = rng.normal(size=(T, dU))
        pol_covar = random_spd((T, dU, dU), rng)
        chol_pol_covar = np.empty_like(pol_covar)
        inv_pol_covar = np.empty_like(pol_covar)

        # Compute Cholesky decomposition and inverse
        for t in range(T):
            chol_pol_covar[t] = np.linalg.cholesky(pol_covar[t])
            inv_pol_covar[t] = np.linalg.inv(pol_covar[t])

        pol = LinearGaussianPolicy(K, k, pol_covar)

        assert_array_almost_equal(K, pol.K)
        assert_array_almost_equal(k, pol.k)
        assert_array_almost_equal(pol_covar, pol.pol_covar)
        assert_array_almost_equal(chol_pol_covar, pol.chol_pol_covar)
        assert_array_almost_equal(inv_pol_covar, pol.inv_pol_covar)

    def test_init_from_inv_pol_covar(self):
        """Test __init__ using inv_pol_covar."""
        from donk.policy import LinearGaussianPolicy

        T, dX, dU = 10, 10, 6
        rng = np.random.default_rng(1)

        K = rng.normal(size=(T, dU, dX))
        k = rng.normal(size=(T, dU))
        pol_covar = random_spd((T, dU, dU), rng)
        chol_pol_covar = np.empty_like(pol_covar)
        inv_pol_covar = np.empty_like(pol_covar)

        # Compute Cholesky decomposition and inverse
        for t in range(T):
            chol_pol_covar[t] = np.linalg.cholesky(pol_covar[t])
            inv_pol_covar[t] = np.linalg.inv(pol_covar[t])

        pol = LinearGaussianPolicy(K, k, inv_pol_covar=inv_pol_covar)

        assert_array_almost_equal(K, pol.K)
        assert_array_almost_equal(k, pol.k)
        assert_array_almost_equal(pol_covar, pol.pol_covar)
        assert_array_almost_equal(chol_pol_covar, pol.chol_pol_covar)
        assert_array_almost_equal(inv_pol_covar, pol.inv_pol_covar)

    def test_act(self):
        """Check act producing proper distribution."""
        from donk.policy import LinearGaussianPolicy

        rnd = np.random.RandomState(0)
        T, dX, dU = 1, 2, 3

        K = np.tile(np.arange(dU * dX).reshape(dU, dX), (T, 1, 1))
        k = np.tile(np.arange(dU), (T, 1))
        pol_covar = np.tile(np.diag(np.arange(dU) + 1.0), (T, 1, 1))

        pol = LinearGaussianPolicy(K, k, pol_covar)

        N = 200
        u = np.empty((N, T, dU))
        for i in range(N):
            u[i] = pol.act(np.ones(dX), 0, noise=rnd.randn(dU))

        assert_array_almost_equal(np.mean(u, axis=0), [[1.031999, 5.962399, 10.705097]])
        assert_array_almost_equal(np.var(u, axis=0), [[1.039798, 2.04722, 2.807949]])


class Test_Noise(unittest.TestCase):

    def test_smooth_noise(self):
        """Test smooth_noise."""
        from donk.policy import smooth_noise

        # Uniform noise
        noise = np.random.default_rng(0).uniform(-1, 1, (20, 4))
        smoothed = smooth_noise(noise, 1)

        assert_array_almost_equal(np.var(smoothed, axis=0), np.var(noise, axis=0))
        assert_array_almost_equal(np.mean(smoothed, axis=0), np.mean(noise, axis=0))

        # Normal noise
        noise = np.random.default_rng(0).normal(0, 1, (10, 3))
        smoothed = smooth_noise(noise, 2)

        assert_array_almost_equal(np.var(smoothed, axis=0), np.var(noise, axis=0))
        assert_array_almost_equal(np.mean(smoothed, axis=0), np.mean(noise, axis=0))


class Test_Initial_Policies(unittest.TestCase):

    def test_constant_policy(self):
        """Test constant_policy."""
        from donk.policy import initial_policies

        T, dX, dU = 10, 5, 2
        pol = initial_policies.constant_policy(T=T, dX=dX, u=[1, 2], variance=[0.5, 0.25])

        for t in range(T):
            with self.subTest(t=t):
                assert_array_almost_equal(pol.K[t], np.zeros((dU, dX)))
                assert_array_almost_equal(pol.k[t], [1, 2])
                assert_array_almost_equal(pol.pol_covar[t], [[0.5, 0], [0, 0.25]])
                assert_array_almost_equal(pol.chol_pol_covar[t], np.sqrt([[0.5, 0], [0, 0.25]]))
                assert_array_almost_equal(pol.inv_pol_covar[t], [[2, 0], [0, 4]])

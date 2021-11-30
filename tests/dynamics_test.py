import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from tests.utils import random_spd


class Test_LinearDynamics(unittest.TestCase):

    def test_predict(self):
        from donk.dynamics import LinearDynamics

        dynamics = LinearDynamics(
            F=np.array([[[1, 0, 1], [0, 1, 1]]]),
            f=np.array([[0, 1]]),
            covar=np.array([[[0.1, 0.1], [0.1, 0.1]]]),
        )

        assert_array_equal(dynamics.predict(x=[1, 1], u=[1], t=0, noise=None), [2, 3])

    def test_fit_lr(self):
        from donk.dynamics.linear_dynamics import fit_lr

        with np.load("tests/data/traj_00.npz") as data:
            X = data['X']
            U = data['U'][:, :-1]
        _, _, dX = X.shape
        _, T, dU = U.shape

        dyn = fit_lr(X, U, regularization=1e-6)
        F, f, dyn_covar = dyn.F, dyn.f, dyn.covar

        # Check shapes
        assert_array_equal(F.shape, (T, dX, dX + dU))
        assert_array_equal(f.shape, (T, dX))
        assert_array_equal(dyn_covar.shape, (T, dX, dX))

        # Check some values
        assert_allclose(
            F[0, -1], [
                0.00000000e+00, -9.98390720e+00, -1.29479403e+01, -1.32099334e+00, -1.61173430e+00, -6.70051884e+01, 2.30815914e+01,
                2.17293139e-01, 0.00000000e+00, 2.69500687e-23, -2.69500687e-23, 0.00000000e+00, -5.39001375e-23, 3.36875859e-24,
                -1.34750344e-23, 4.56497288e-01, -2.70607228e-01
            ]
        )
        assert_allclose(
            F[-1, 0], [
                4.14986607e-02, -4.61995618e-01, -1.18756810e+00, 3.43210881e-01, 5.92291666e-01, -3.09607786e-01, -4.69028078e-02,
                -2.30810980e-01, 3.50226060e-03, -1.19272129e-01, 9.78285647e-02, 6.28585341e-01, 4.32602881e-01, -8.24673225e-13,
                7.16121964e-03, 9.89241563e-03, 6.08668508e-01
            ]
        )
        # Check s.p.d.
        assert_array_equal(dyn_covar, np.swapaxes(dyn_covar, 1, 2), "dyn_covar not symmetric")
        for t in range(T):
            with self.subTest(t=t):
                self.assertTrue(all(np.linalg.eigvalsh(dyn_covar[t]) >= -1e-16), f"Negative eigenvalues {np.linalg.eigvalsh(dyn_covar[t])}")

    def test_fit_lr_error(self):
        from donk.dynamics.linear_dynamics import fit_lr

        N, T, dX, dU = 1, 4, 3, 2
        X = np.empty((N, T + 1, dX))
        U = np.empty((N, T, dU))

        with self.assertRaises(ValueError):
            fit_lr(X, U, regularization=1e-6)

    def test_fit_lr_with_prior(self):
        from donk.dynamics import to_transitions
        from donk.dynamics.linear_dynamics import fit_lr
        from donk.dynamics.prior import GMMPrior

        with np.load("tests/data/traj_00.npz") as data:
            X = data['X']
            U = data['U'][:, :-1]
        _, _, dX = X.shape
        _, T, dU = U.shape

        prior = GMMPrior(8, random_state=0)
        prior.update(to_transitions(X, U).reshape(-1, dX + dU + dX))

        dyn = fit_lr(X[:3], U[:3], prior, regularization=0)
        F, f, dyn_covar = dyn.F, dyn.f, dyn.covar

        # Check shapes
        assert_array_equal(F.shape, (T, dX, dX + dU))
        assert_array_equal(f.shape, (T, dX))
        assert_array_equal(dyn_covar.shape, (T, dX, dX))

        # Check some values
        assert_allclose(
            F[0, -1], [
                6.4670984068e-12, -9.4600223921e+00, -1.2259576378e+01, -1.2686138915e+00, -1.7659779482e+00, -3.6820406902e+01,
                2.8851768774e+01, 2.2697946776e-01, -3.2335492034e-12, -6.4670984068e-12, 3.2335492034e-12, -6.4670984068e-12,
                0.0000000000e+00, 0.0000000000e+00, -1.6167746017e-12, 4.9092612094e-01, -2.3294766657e-01
            ],
            atol=1e-8
        )
        assert_allclose(
            F[-1, 0], [
                5.2810630480e-02, 1.1964354976e-02, -1.4035005294e-01, -1.0247787101e-02, -6.3580697713e-02, 1.1825430983e-02,
                7.3951825793e-03, 3.5634195593e-03, 8.7252680504e-03, 2.0284214112e-02, -3.2658321456e-02, -1.9436559723e-02,
                1.8730195626e-01, 2.6472698883e-02, 5.9203433330e-03, -2.0699574315e-04, 5.4962668583e-01
            ],
            atol=1e-8
        )
        # Check s.p.d.
        assert_array_equal(dyn_covar, np.swapaxes(dyn_covar, 1, 2), "dyn_covar not symmetric")
        for t in range(T):
            with self.subTest(t=t):
                self.assertTrue(all(np.linalg.eigvalsh(dyn_covar[t]) >= 0), f"Negative eigenvalues {np.linalg.eigvalsh(dyn_covar[t])}")

    def test_log_prob(self):
        from donk.dynamics import to_transitions
        from donk.dynamics.linear_dynamics import fit_lr
        from donk.dynamics.prior import GMMPrior

        with np.load("tests/data/traj_00.npz") as data:
            X = data['X']
            U = data['U'][:, :-1]
        N, _, dX = X.shape
        _, T, dU = U.shape

        X_train, U_train = X[:-3], U[:-3]
        X_test, U_test = X[-3:], U[-3:]

        prior = GMMPrior(8, random_state=0)
        prior.update(to_transitions(X_train, U_train).reshape(-1, dX + dU + dX))

        dyn = fit_lr(X_train[:], U_train[:], prior, regularization=0)

        log_prob = dyn.log_prob(X_test, U_test)

        # Check shapes
        assert_array_equal(log_prob.shape, (3, T))

        assert_allclose(np.mean(log_prob, axis=-1), [-109.7484287601, -98.2677612391, -6554.897500492])

    def test_str(self):
        """Test LinearDynamics.__str__."""
        from donk.dynamics import LinearDynamics

        T, dX, dU = 4, 3, 2

        dyn = LinearDynamics(
            np.empty((T, dX, dX + dU)),
            np.empty((T, dX)),
            np.empty((T, dX, dX)),
        )

        self.assertEqual(str(dyn), "LinearDynamics[T=4, dX=3, dU=2]")


class Test_NormalInverseWishart(unittest.TestCase):

    def test_prior_map(self):
        from donk.dynamics.prior import NormalInverseWishart

        emp_mean = np.array([1])
        emp_covar = np.array([[0.5]])
        N = 1

        prior = NormalInverseWishart.non_informative_prior(1).posterior(emp_mean, emp_covar, N)

        assert_allclose(prior.map_mean(), [1])
        assert_allclose(prior.map_covar(), [[0.5]])

    def test_posterior_order_invariance(self):
        from donk.dynamics.prior import NormalInverseWishart

        d = 10
        rng = np.random.default_rng(0)

        # Sample set 1
        emp_mean_1 = np.array(rng.standard_normal(d))
        emp_covar_1 = random_spd((d, d), rng)
        N_1 = 42

        # Sample set 1
        emp_mean_2 = np.array(rng.standard_normal(d))
        emp_covar_2 = random_spd((d, d), rng)
        N_2 = 69

        prior = NormalInverseWishart.non_informative_prior(d)

        posterior_1 = prior.posterior(emp_mean_1, emp_covar_1, N_1).posterior(emp_mean_2, emp_covar_2, N_2)
        # Reverse application order
        posterior_2 = prior.posterior(emp_mean_2, emp_covar_2, N_2).posterior(emp_mean_1, emp_covar_1, N_1)

        assert_allclose(posterior_1.mu0, posterior_2.mu0)
        assert_allclose(posterior_1.Phi, posterior_2.Phi)
        assert_allclose(posterior_1.N_mean, posterior_2.N_mean)
        assert_allclose(posterior_1.N_covar, posterior_2.N_covar)
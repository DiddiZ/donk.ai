import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal


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
        from donk.dynamics.linear_dynamics import fit_lr
        from donk.dynamics.prior import GMMPrior
        from donk.dynamics import to_transitions

        with np.load("tests/data/traj_00.npz") as data:
            X = data['X']
            U = data['U'][:, :-1]
        _, _, dX = X.shape
        _, T, dU = U.shape

        prior = GMMPrior(8, random_state=0)
        prior.update(to_transitions(X, U).reshape(-1, dX + dU + dX))

        dyn = fit_lr(X[:3], U[:3], prior, regularization=1e-6)
        F, f, dyn_covar = dyn.F, dyn.f, dyn.covar

        # Check shapes
        assert_array_equal(F.shape, (T, dX, dX + dU))
        assert_array_equal(f.shape, (T, dX))
        assert_array_equal(dyn_covar.shape, (T, dX, dX))

        # Check some values
        assert_allclose(
            F[0, -1], [
                0.9403596673, -8.2294893523, -10.8861634988, -1.1014404065, -1.7100888584, 4.1934859476, 10.5373063746, 0.2593118824,
                0.9403596719, 0.9403596719, 0.9403596719, 0.9403596719, 0.9403596719, 0.9403596719, 0.9403596719, 0.5831041918,
                -0.1689043096
            ]
        )
        assert_allclose(
            F[-1, 0],
            [
                0.0531501627, 0.0126418035, -0.1449997338, -0.0114061895, -0.0562531154, 0.0083263427, 0.0082598694, 0.0031456036,
                0.0086516918, 0.0240835597, -0.0428903516, 0.0172597417, 0.1589828755, 0.4074868557, 0.0057786167, -0.0009862849,
                0.5487683864
            ],
        )
        # Check s.p.d.
        assert_array_equal(dyn_covar, np.swapaxes(dyn_covar, 1, 2), "dyn_covar not symmetric")
        for t in range(T):
            with self.subTest(t=t):
                self.assertTrue(all(np.linalg.eigvalsh(dyn_covar[t]) >= 0), f"Negative eigenvalues {np.linalg.eigvalsh(dyn_covar[t])}")

    def test_log_prob(self):
        from donk.dynamics.linear_dynamics import fit_lr
        from donk.dynamics.prior import GMMPrior
        from donk.dynamics import to_transitions

        with np.load("tests/data/traj_00.npz") as data:
            X = data['X']
            U = data['U'][:, :-1]
        N, _, dX = X.shape
        _, T, dU = U.shape

        X_train, U_train = X[:-3], U[:-3]
        X_test, U_test = X[-3:], U[-3:]

        prior = GMMPrior(8, random_state=0)
        prior.update(to_transitions(X_train, U_train).reshape(-1, dX + dU + dX))

        dyn = fit_lr(X_train[:], U_train[:], prior, regularization=1e-6)

        log_prob = dyn.log_prob(X_test, U_test)

        # Check shapes
        assert_array_equal(log_prob.shape, (3, T))

        assert_allclose(np.mean(log_prob, axis=-1), [-5282.53359794, -5765.19471819, -3072937.55992457])

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

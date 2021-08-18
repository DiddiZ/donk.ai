import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal


class Test_LinearDynamics(unittest.TestCase):

    def test_predict(self):
        from donk.dynamics import LinearDynamics

        dynamics = LinearDynamics(
            Fm=np.array([[[1, 0, 1], [0, 1, 1]]]),
            fv=np.array([[0, 1]]),
            dyn_covar=np.array([[[0.1, 0.1], [0.1, 0.1]]]),
        )

        assert_array_equal(dynamics.predict(x=[1, 1], u=[1], t=0, noise=None), [2, 3])

    def test_fit_lr(self):
        from donk.dynamics.linear_dynamics import fit_lr

        with np.load("tests/data/traj_00.npz") as data:
            X = data['X']
            U = data['U'][:, :-1]
        N, _, dX = X.shape
        _, T, dU = U.shape

        dyn = fit_lr(X, U, regularization=1e-6)
        Fm, fv, dyn_covar = dyn.Fm, dyn.fv, dyn.dyn_covar

        # Check shapes
        assert_array_equal(Fm.shape, (T, dX, dX + dU))
        assert_array_equal(fv.shape, (T, dX))
        assert_array_equal(dyn_covar.shape, (T, dX, dX))

        # Check some values
        assert_allclose(
            Fm[0, -1], [
                0.00000000e+00, -9.98390720e+00, -1.29479403e+01, -1.32099334e+00, -1.61173430e+00, -6.70051884e+01, 2.30815914e+01,
                2.17293139e-01, 0.00000000e+00, 2.69500687e-23, -2.69500687e-23, 0.00000000e+00, -5.39001375e-23, 3.36875859e-24,
                -1.34750344e-23, 4.56497288e-01, -2.70607228e-01
            ]
        )
        assert_allclose(
            Fm[-1, 0], [
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

    def test_fit_lr_with_prior(self):
        from donk.dynamics.linear_dynamics import fit_lr
        from donk.dynamics.prior import GMMPrior

        with np.load("tests/data/traj_00.npz") as data:
            X = data['X']
            U = data['U'][:, :-1]
        N, _, dX = X.shape
        _, T, dU = U.shape
        transitions = np.c_[X[:, :-1], U, X[:, 1:]].reshape(N * T, dX + dU + dX)

        prior = GMMPrior(8, random_state=0)
        prior.update(transitions)

        dyn = fit_lr(X[:3], U[:3], prior, regularization=1e-6)
        Fm, fv, dyn_covar = dyn.Fm, dyn.fv, dyn.dyn_covar

        # Check shapes
        assert_array_equal(Fm.shape, (T, dX, dX + dU))
        assert_array_equal(fv.shape, (T, dX))
        assert_array_equal(dyn_covar.shape, (T, dX, dX))

        # Check some values
        assert_allclose(
            Fm[0, -1], [
                1.1901108606, -8.2155541561, -10.8582036973, -1.0970622672, -1.6993010666, 3.8033234782, 8.8718137698, 0.2596884979,
                1.1901108592, 1.1901108596, 1.1901108597, 1.1901108597, 1.1901108598, 1.1901108598, 1.1901108591, 0.5832087672,
                -0.1688012448
            ]
        )
        assert_allclose(
            Fm[-1, 0],
            [
                0.0531532012, 0.0126352773, -0.1446274682, -0.0114229692, -0.0562042267, 0.0083028760, 0.0082817474, 0.0031557044,
                0.0086551089, 0.0240643630, -0.0428469319, 0.0176277952, 0.1582171429, 0.4074299145, 0.0057864226, -0.0009779524,
                0.5487691228
            ],
        )
        # Check s.p.d.
        assert_array_equal(dyn_covar, np.swapaxes(dyn_covar, 1, 2), "dyn_covar not symmetric")
        for t in range(T):
            with self.subTest(t=t):
                self.assertTrue(all(np.linalg.eigvalsh(dyn_covar[t]) >= 0), f"Negative eigenvalues {np.linalg.eigvalsh(dyn_covar[t])}")

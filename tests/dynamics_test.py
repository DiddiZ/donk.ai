import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal


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

        Fm, fv, dyn_covar = fit_lr(X, U, regularization=1e-6)

        # Check shapes
        assert_array_equal(Fm.shape, (T, dX, dX + dU))
        assert_array_equal(fv.shape, (T, dX))
        assert_array_equal(dyn_covar.shape, (T, dX, dX))

        # Check some values
        assert_array_almost_equal(
            Fm[0, -1], [
                0.00000000e+00, -9.98390720e+00, -1.29479403e+01, -1.32099334e+00, -1.61173430e+00, -6.70051884e+01,
                2.30815914e+01, 2.17293139e-01, 0.00000000e+00, 2.69500687e-23, -2.69500687e-23, 0.00000000e+00,
                -5.39001375e-23, 3.36875859e-24, -1.34750344e-23, 4.56497288e-01, -2.70607228e-01
            ]
        )
        assert_array_almost_equal(
            Fm[-1, 0], [
                4.14986607e-02, -4.61995618e-01, -1.18756810e+00, 3.43210881e-01, 5.92291666e-01, -3.09607786e-01,
                -4.69028078e-02, -2.30810980e-01, 3.50226060e-03, -1.19272129e-01, 9.78285647e-02, 6.28585341e-01,
                4.32602881e-01, -8.24673225e-13, 7.16121964e-03, 9.89241563e-03, 6.08668508e-01
            ]
        )

    def test_fit_lr_with_prior(self):
        from donk.dynamics.linear_dynamics import fit_lr
        from donk.dynamics.prior import GMMPrior

        with np.load("tests/data/traj_00.npz") as data:
            X = data['X']
            U = data['U'][:, :-1]
        N, _, dX = X.shape
        _, T, dU = U.shape
        transitions = np.c_[X[:, :-1], U, X[:, 1:]].reshape(N * T, dX + dU + dX)

        prior = GMMPrior(max_clusters=8, random_state=0)
        prior.update(transitions)

        Fm, fv, dyn_covar = fit_lr(X[:3], U[:3], regularization=1e-6)

        # Check shapes
        assert_array_equal(Fm.shape, (T, dX, dX + dU))
        assert_array_equal(fv.shape, (T, dX))
        assert_array_equal(dyn_covar.shape, (T, dX, dX))

        # Check some values
        assert_array_almost_equal(
            Fm[0, -1], [
                0.00000000e+00, -1.59097886e-03, -1.55670197e-03, 3.66813669e-03, -5.80934222e-06, 2.65460765e-05,
                -4.26546524e-06, 1.35521681e-01, 0.00000000e+00, 3.62010170e-26, -3.62010170e-26, 0.00000000e+00,
                0.00000000e+00, 0.00000000e+00, 1.81005085e-26, -5.38810305e-02, 4.05879352e-02
            ]
        )
        assert_array_almost_equal(
            Fm[-1, 0], [
                -0.01492868, -0.05346967, 0.02899837, 0.00083401, -0.02374431, -0.00567905, 0.00396877, 0.03335041,
                0.05123336, 0.03796068, 0.04711648, 0.02343822, 0.02809063, 0., -0.01798136, 0.0703149, 0.03685244
            ]
        )

import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_less


class Test_LinearDynamics(unittest.TestCase):

    def test_fit_lr(self):
        """Test __init__ using pol_covar."""
        from donk.dynamics.linear_dynamics import fit_lr

        with np.load("tests/data/traj_00.npz") as data:
            X = data['X']
            U = data['U']

        Fm, fv, dyn_covar = fit_lr(X, U, regularization=1e-6)

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

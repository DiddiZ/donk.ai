import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal


class Test_LQG(unittest.TestCase):

    def test_forward(self):
        from donk.traj_opt import lqg
        from tests.utils import random_spd, random_tvlg, random_lq_pol

        T, dX, dU = 5, 3, 2
        rng = np.random.default_rng(0)

        x0_mean = rng.normal(size=(dX))
        x0_covar = random_spd((dX, dX), rng)
        dyn = random_tvlg(T, dX, dU, rng)
        pol = random_lq_pol(T, dX, dU, rng)
        traj_mean, traj_covar = lqg.forward(dyn, pol, x0_mean, x0_covar)

        # Check shapes
        assert_array_equal(traj_mean.shape, (T, dX + dU))
        assert_array_equal(traj_covar.shape, (T, dX + dU, dX + dU))

        # Check some values
        assert_array_almost_equal(
            traj_mean, [
                [0.12573022, -0.13210486, 0.64042265, -2.21332089, 1.2562705], [1.17023217, 1.8200327, 0.9955234, -3.87002688, -1.53651116],
                [2.24876315, 2.2154459, 4.87559943, -5.68264397, -4.03765651],
                [1.20832951, 17.90919574, -7.78669286, 1.24165172, 20.36111693],
                [-39.41244121, 21.99347233, 42.74031851, -2.26076952, -7.1186971]
            ]
        )
        assert_array_almost_equal(
            traj_covar[3:], [
                [
                    [54.09235666, -48.4300889, 50.45096717, 105.0393554, -113.93189734],
                    [-48.4300889, 423.56745, -273.31189737, -132.74023589, 519.01789733],
                    [50.45096717, -273.31189737, 192.11552154, 121.89507028, -348.99794792],
                    [105.0393554, -132.74023589, 121.89507028, 210.15802318, -262.59827268],
                    [-113.93189734, 519.01789733, -348.99794792, -262.59827268, 706.79975636]
                ],
                [
                    [2004.44519505, -1054.311996, -2452.1319988, -159.40800433, 470.24722393],
                    [-1054.311996, 568.21099685, 1271.69753311, 67.91357455, -236.21633069],
                    [-2452.1319988, 1271.69753311, 3174.01281199, 353.83421056, -612.88196643],
                    [-159.40800433, 67.91357455, 353.83421056, 160.10573042, -70.82176365],
                    [470.24722393, -236.21633069, -612.88196643, -70.82176365, 125.36773789]
                ]
            ]
        )
        assert_array_equal(traj_covar, np.swapaxes(traj_covar, 1, 2), "traj_covar not symmetric")

    def test_backward(self):
        from donk.traj_opt import lqg
        from tests.utils import random_spd, random_tvlg, random_lq_pol
        from donk.costs import loss_l2

        T, dX, dU = 5, 3, 2
        rng = np.random.default_rng(0)

        dyn = random_tvlg(T, dX, dU, rng)
        _, c, C = loss_l2(
            x=rng.normal(size=(T + 1, dX + dU)),
            t=rng.normal(size=(T + 1, dX + dU)),
            w=np.concatenate([rng.normal(size=(T + 1, dX)) > 0, 1e-2 * np.ones((T + 1, dU))], axis=1)
        )
        pol = lqg.backward(dyn, C, c)

        # Check some values
        assert_array_almost_equal(
            pol.K, [
                [[1.30823423, 1.38208415, 4.50630981], [-0.71211717, -0.19657582, -2.59400007]],
                [[0.84046166, 1.61337651, -1.2593433], [0.12026023, -0.50799617, 1.20987989]],
                [[2.1477244, -1.44199314, -1.17861916], [-1.37101699, 0.98956977, 0.52860322]],
                [[-0.04972842, -0.23628132, -1.02031922], [0.3636096, -0.50788773, 0.35318357]],
                [[-1.11351665, -1.82347015, 4.77470105], [-0.42326492, -0.27333697, 1.6068229]]
            ]
        )

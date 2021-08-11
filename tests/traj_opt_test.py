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
        assert_array_equal(traj_mean.shape, (T + 1, dX + dU))
        assert_array_equal(traj_covar.shape, (T + 1, dX + dU, dX + dU))

        # Check some values
        assert_array_almost_equal(
            traj_mean, [
                [0.12573022, -0.13210486, 0.64042265, -2.21332089, 1.2562705],
                [1.17023217, 1.8200327, 0.9955234, -3.87002688, -1.53651116],
                [2.24876315, 2.2154459, 4.87559943, -5.68264397, -4.03765651],
                [1.20832951, 17.90919574, -7.78669286, 1.24165172, 20.36111693],
                [-39.41244121, 21.99347233, 42.74031851, -2.26076952, -7.1186971],
                [31.0077354, 54.39317675, -68.29849519, 0, 0],
            ]
        )
        assert_array_almost_equal(
            traj_covar[3:], [
                [
                    [54.09235666, -48.4300889, 50.45096717, 105.0393554, -113.93189734],
                    [-48.4300889, 423.56745, -273.31189737, -132.74023589, 519.01789733],
                    [50.45096717, -273.31189737, 192.11552154, 121.89507028, -348.99794792],
                    [105.0393554, -132.74023589, 121.89507028, 210.15802318, -262.59827268],
                    [-113.93189734, 519.01789733, -348.99794792, -262.59827268, 706.79975636],
                ],
                [
                    [2004.44519505, -1054.311996, -2452.1319988, -159.40800433, 470.24722393],
                    [-1054.311996, 568.21099685, 1271.69753311, 67.91357455, -236.21633069],
                    [-2452.1319988, 1271.69753311, 3174.01281199, 353.83421056, -612.88196643],
                    [-159.40800433, 67.91357455, 353.83421056, 160.10573042, -70.82176365],
                    [470.24722393, -236.21633069, -612.88196643, -70.82176365, 125.36773789],
                ],
                [
                    [1952.24270273, 3582.01507711, -3612.1454295, 0, 0],
                    [3582.01507711, 6751.46081088, -6587.92373881, 0, 0],
                    [-3612.1454295, -6587.92373881, 6928.35963809, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ]
            ]
        )
        assert_array_equal(traj_covar, np.swapaxes(traj_covar, 1, 2), "traj_covar not symmetric")

    def test_backward(self):
        from donk.traj_opt import lqg
        from tests.utils import random_tvlg
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

    def test_extended_cost(self):
        from donk.traj_opt import lqg
        from tests.utils import random_lq_pol

        T, dX, dU = 5, 2, 3
        rng = np.random.default_rng(0)

        prev_pol = random_lq_pol(T, dX, dU, rng)
        C, c = lqg.extended_costs_kl(prev_pol)

        assert_array_almost_equal(
            C[:2], [
                [
                    [0.27488267, -0.07112635, -0.05038193, -0.21743656, 0.24137434],
                    [-0.07112635, 0.05152253, 0.04783474, -0.00232771, -0.12433563],
                    [-0.05038193, 0.04783474, 0.26116107, -0.00277339, -0.0360712],
                    [-0.21743656, -0.00232771, -0.00277339, 0.2773116, -0.07502507],
                    [0.24137434, -0.12433563, -0.0360712, -0.07502507, 0.35244007],
                ],
                [
                    [0.95586832, 0.77652347, -0.46259588, 0.27573695, 0.25445846],
                    [0.77652347, 0.88911419, -0.34878502, 0.44213479, 0.01694592],
                    [-0.46259588, -0.34878502, 0.31127984, -0.04400365, -0.04126599],
                    [0.27573695, 0.44213479, -0.04400365, 0.31624371, -0.00673162],
                    [0.25445846, 0.01694592, -0.04126599, -0.00673162, 0.32952559],
                ]
            ]
        )
        assert_array_almost_equal(
            c, [
                [-0.0579161, 0.02801059, 0.25734941, 0.04326088, 0.00400572],
                [0.1005759, 0.08770845, -0.14424373, -0.04169318, -0.09334087],
                [0.41123218, 0.00863011, 0.20377474, 0.04849396, -0.22593934],
                [-0.02510049, -0.09115981, -0.45016499, 0.51723102, -0.41624546],
                [0.22628684, -0.12542044, -0.37047029, -0.16602141, 0.03280298],
            ]
        )

    def test_kl_divergence_action(self):
        from donk.traj_opt import lqg
        from tests.utils import random_lq_pol

        T, dX, dU = 5, 3, 2
        rng = np.random.default_rng(0)
        X = rng.standard_normal((T, dX))
        pol = random_lq_pol(T, dX, dU, rng)
        prev_pol = random_lq_pol(T, dX, dU, rng)

        kl_div = lqg.kl_divergence_action(X, pol, prev_pol)

        self.assertAlmostEqual(kl_div, 3.914744335914712)

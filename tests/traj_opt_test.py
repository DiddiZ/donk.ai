import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal


class Test_LQG(unittest.TestCase):
    def test_forward(self):
        from donk.samples import StateDistribution
        from donk.traj_opt import lqg
        from tests.utils import random_lq_pol, random_spd, random_tvlg

        T, dX, dU = 5, 3, 2
        rng = np.random.default_rng(0)

        x0 = StateDistribution(mean=rng.normal(size=(dX)), covar=random_spd((dX, dX), rng))
        dyn = random_tvlg(T, dX, dU, rng)
        pol = random_lq_pol(T, dX, dU, rng)
        traj = lqg.forward(dyn, pol, x0)

        # Check shapes
        assert_array_equal(traj.mean.shape, (T + 1, dX + dU))
        assert_array_equal(traj.covar.shape, (T + 1, dX + dU, dX + dU))

        # Check some values
        assert_allclose(
            traj.mean,
            [
                [0.12573022, -0.13210486, 0.64042265, -2.21332089, 1.2562705],
                [1.17023217, 1.8200327, 0.9955234, -3.87002688, -1.53651116],
                [2.24876315, 2.2154459, 4.87559943, -5.68264397, -4.03765651],
                [1.20832951, 17.90919574, -7.78669286, 1.24165172, 20.36111693],
                [-39.41244121, 21.99347233, 42.74031851, -2.26076952, -7.1186971],
                [31.0077354, 54.39317675, -68.29849519, 0, 0],
            ],
        )
        assert_allclose(
            traj.covar[3:],
            [
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
                ],
            ],
        )
        assert_array_equal(traj.covar, np.swapaxes(traj.covar, 1, 2), "traj_covar not symmetric")

    def test_backward(self):
        from donk.costs import loss_l2
        from donk.traj_opt import lqg
        from tests.utils import random_tvlg

        T, dX, dU = 5, 3, 2
        rng = np.random.default_rng(0)

        dyn = random_tvlg(T, dX, dU, rng)
        _, c, C = loss_l2(
            x=rng.normal(size=(T + 1, dX + dU)),
            t=rng.normal(size=(T + 1, dX + dU)),
            w=np.concatenate([rng.normal(size=(T + 1, dX)) > 0, 1e-2 * np.ones((T + 1, dU))], axis=1),
        )
        pol = lqg.backward(dyn, C, c)

        # Check some values
        assert_allclose(
            pol.K,
            [
                [[1.30823423, 1.38208415, 4.50630981], [-0.71211717, -0.19657582, -2.59400007]],
                [[0.84046166, 1.61337651, -1.2593433], [0.12026023, -0.50799617, 1.20987989]],
                [[2.1477244, -1.44199314, -1.17861916], [-1.37101699, 0.98956977, 0.52860322]],
                [[-0.04972842, -0.23628132, -1.02031922], [0.3636096, -0.50788773, 0.35318357]],
                [[-1.11351665, -1.82347015, 4.77470105], [-0.42326492, -0.27333697, 1.6068229]],
            ],
        )

    def test_extended_cost(self):
        from donk.traj_opt import lqg
        from tests.utils import random_lq_pol

        T, dX, dU = 5, 2, 3
        rng = np.random.default_rng(0)

        prev_pol = random_lq_pol(T, dX, dU, rng)
        C, c = lqg.extended_costs_kl(prev_pol)

        assert_allclose(
            C[:2],
            [
                [
                    [0.2748826738, -0.0711263456, -0.0503819346, -0.2174365634, 0.2413743407],
                    [-0.0711263456, 0.0515225272, 0.0478347446, -0.0023277054, -0.1243356279],
                    [-0.0503819346, 0.0478347446, 0.2611610740, -0.0027733907, -0.0360712003],
                    [-0.2174365634, -0.0023277054, -0.0027733907, 0.2773116042, -0.0750250658],
                    [0.2413743407, -0.1243356279, -0.0360712003, -0.0750250658, 0.3524400661],
                ],
                [
                    [0.9558683162, 0.7765234677, -0.4625958804, 0.2757369530, 0.2544584564],
                    [0.7765234677, 0.8891141945, -0.3487850167, 0.4421347864, 0.0169459245],
                    [-0.4625958804, -0.3487850167, 0.3112798376, -0.0440036518, -0.0412659902],
                    [0.2757369530, 0.4421347864, -0.0440036518, 0.3162437051, -0.0067316240],
                    [0.2544584564, 0.0169459245, -0.0412659902, -0.0067316240, 0.3295255889],
                ],
            ],
        )
        assert_allclose(
            c,
            [
                [-0.0579161036, 0.0280105873, 0.2573494063, 0.0432608817, 0.0040057220],
                [0.1005758960, 0.0877084519, -0.1442437337, -0.0416931800, -0.0933408681],
                [0.4112321773, 0.0086301113, 0.2037747362, 0.0484939612, -0.2259393350],
                [-0.0251004913, -0.0911598106, -0.4501649890, 0.5172310245, -0.4162454596],
                [0.2262868408, -0.1254204382, -0.3704702903, -0.1660214078, 0.0328029829],
            ],
        ),

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


class TrajectoryDistribution(unittest.TestCase):
    def test_trajectory_distribution(self):
        from donk.samples import TrajectoryDistribution
        from tests.utils import random_spd

        T, dX, dU = 5, 3, 2
        rng = np.random.default_rng(0)

        mean = rng.standard_normal((T, dX + dU))
        covar = random_spd((T, dX + dU, dX + dU), rng)
        traj = TrajectoryDistribution(mean, covar, dX=dX)

        assert_array_equal(traj.mean, mean)
        assert_array_equal(traj.covar, covar)
        assert_array_equal(traj.X_mean, mean[:, :dX])
        assert_array_equal(traj.U_mean, mean[:-1, dX:])
        assert_array_equal(traj.X_covar, covar[:, :dX, :dX])
        assert_array_equal(traj.U_covar, covar[:-1, dX:, dX:])

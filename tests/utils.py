"""Utilities for unit testing."""
import numpy as np


def random_spd(dX):
    """Compute a random matric guaranteeed to be s.p.d."""
    rnd = np.random.rand(dX, dX)
    return 0.5 * (rnd + rnd.T) + np.eye(dX) * dX

"""Utilities for unit testing."""

import numpy as np

from donk.dynamics import LinearDynamics
from donk.policy import LinearGaussianPolicy
from donk.utils import symmetrize


def random_spd(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """Generate an arbitrarily random matrix guaranteeed to be s.p.d."""
    return symmetrize(rng.uniform(size=shape)) + np.eye(shape[-1]) * shape[-1]


def random_lq_pol(T: int, dX: int, dU: int, rng: np.random.Generator) -> LinearGaussianPolicy:
    """Generate an arbitrarily random linear gaussian policy."""
    K = rng.normal(size=(T, dU, dX))
    k = rng.normal(size=(T, dU))
    pol_covar = random_spd((T, dU, dU), rng)
    return LinearGaussianPolicy(K, k, pol_covar)


def random_tvlg(T: int, dX: int, dU: int, rng: np.random.Generator) -> LinearDynamics:
    """Generate arbitrarily random linear gaussian dynamics."""
    Fm = rng.normal(size=(T, dX, dX + dU))
    fv = rng.normal(size=(T, dX))
    dyn_covar = random_spd((T, dX, dX), rng)
    return LinearDynamics(Fm, fv, dyn_covar)

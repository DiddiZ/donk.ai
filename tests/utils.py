"""Utilities for unit testing."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from donk.dynamics import LinearDynamics
from donk.policy import LinearGaussianPolicy
from donk.utils import symmetrize


def random_spd(shape: tuple[int], rng: np.random.Generator):
    """Generate an arbitrarily random matrix guaranteeed to be s.p.d."""
    return symmetrize(rng.uniform(size=shape)) + np.eye(shape[-1]) * shape[-1]


def random_lq_pol(T: int, dX: int, dU: int, rng: np.random.Generator):
    """Generate an arbitrarily random linear gaussian policy."""
    K = rng.normal(size=(T, dU, dX))
    k = rng.normal(size=(T, dU))
    pol_covar = random_spd((T, dU, dU), rng)
    return LinearGaussianPolicy(K, k, pol_covar)


def random_tvlg(T: int, dX: int, dU: int, rng: np.random.Generator):
    """Generate arbitrarily random linear gaussian dynamics."""
    Fm = rng.normal(size=(T, dX, dX + dU))
    fv = rng.normal(size=(T, dX))
    dyn_covar = random_spd((T, dX, dX), rng)
    return LinearDynamics(Fm, fv, dyn_covar)


def load_state_controller_dataset(
    dataset: int, itr: int
) -> tuple[np.ndarray, LinearGaussianPolicy, np.ndarray, np.ndarray]:
    """Loads a state_controller dataset.

    Args:
        dataset: Id of the dataset
        itr: Iteration to return

    Returns:
        X: (N, T, dX) Real states
        pol: Fitted linear policy
        X_mean: (T, dX) Mean of state distribution
        X_covar: (T, dX, dX) Covariance of state distribution
    """
    file = Path(f"tests/data/state_controller_{dataset:02d}.npz")
    if not file.is_file():
        raise ValueError(f"There is no dataset 'state_controller_{dataset:02d}.npz'")

    with np.load(file) as data:
        if itr not in range(len(data["X"])):
            raise ValueError(f"Invalid iteration {itr}")

        X = data["X"][itr]
        pol = LinearGaussianPolicy(K=data["K"][itr], k=data["k"][itr], pol_covar=data["pol_covar"][itr])
        X_mean = data["X_mean"][itr]
        X_covar = data["X_covar"][itr]

    return X, pol, X_mean, X_covar

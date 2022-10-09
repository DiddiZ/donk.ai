from pathlib import Path

import numpy as np

from donk.policy import LinearGaussianPolicy


def load_state_controller_dataset(dataset, itr):
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

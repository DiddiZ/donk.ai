import numpy as np


def to_transitions(X: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Convert state-action sequences to state-action-state tuples.

    Args:
        X: (N, T+1, dX), States
        U: (N, T, dU), Actions

    Returns:
        transitions: (N*T, dX+dU+dX)
    """
    N, _, dX = X.shape
    _, T, dU = U.shape
    return np.c_[X[:, :-1], U, X[:, 1:]].reshape(N * T, dX + dU + dX)

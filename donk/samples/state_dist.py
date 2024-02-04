from __future__ import annotations

import numpy as np


class StateDistribution:
    """A normally distributed state."""

    def __init__(self, mean: np.ndarray, covar: np.ndarray) -> None:
        """Initialize this `StateDistribution`.

        Args:
            mean: (..., dX) Means of the state distribution
            covar: (..., dX, dX) Covariances of the state distribution
            dX: Dimension of state space
        """
        dX = mean.shape[-1]

        # Check shapes
        assert covar.shape[-2:] == (dX, dX), f"{covar.shape[-2:]} != {(dX, dX)}"
        assert mean.shape[:-1] == covar.shape[:-2], f"{mean.shape[:-1]} != {covar.shape[:-2]}"

        self.mean = mean
        self.covar = covar
        self.dX = dX

    @staticmethod
    def fit(X: np.ndarray) -> StateDistribution:
        """Fit a `StateDistribution` to some states.

        Args:
            X: (N, dX), states
        """
        return StateDistribution(
            mean=np.mean(X, axis=0),
            covar=np.cov(X, rowvar=False),
        )

from typing import Tuple

import numpy as np


class TrajectoryDistribution:
    """A normally distributed trajectory."""

    def __init__(self, mean: np.ndarray, covar: np.ndarray, dX: int, dU: int = None) -> None:
        """Initialize this `TrajectoryDistribution`.

        Args:
            mean: (..., T, dX+dU) Means of the trajectory distribution
            covar: (..., T, dX+dU, dX+dU) Covariances of the trajectory distribution
            dX: Dimension of state space
            dU: Dimension of action space, may be inferred if not explicitly stated
        """
        if dU is None:
            dU = mean.shape[-1] - dX

        # Check shapes
        assert mean.shape[-1:] == (dX + dU,), f"{mean.shape[-1:]} != {(dX + dU,)}"
        assert covar.shape[-2:] == (dX + dU, dX + dU), f"{covar.shape[-2:]} != {(dX + dU, dX + dU)}"
        assert mean.shape[:-1] == covar.shape[:-2], f"{mean.shape[:-1]} != {covar.shape[:-2]}"

        self.mean = mean
        self.covar = covar
        self.dX = dX
        self.dU = dU

    @property
    def X_mean(self):
        """Get state component of trajectory means."""
        return self.mean[..., : self.dX]

    @property
    def X_covar(self):
        """Get state component of trajectory covariances."""
        return self.covar[..., : self.dX, : self.dX]

    @property
    def U_mean(self):
        """Get action component of trajectory means."""
        return self.mean[..., :-1, self.dX :]

    @property
    def U_covar(self):
        """Get action component of trajectory covariances."""
        return self.covar[..., :-1, self.dX :, self.dX :]

    def sample(self, size: Tuple[int], rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """Draw samples from this trajectory distribution.

        Args:
            size: Shape of desired amount of samples
        """
        T = self.mean.shape[-2] - 1
        N = np.prod(size)

        X = np.empty((N, T + 1, self.dX))
        U = np.empty((N, T, self.dU))

        for t in range(T):
            xu = rng.multivariate_normal(self.mean[t], self.covar[t], size=N)
            X[:, t] = xu[:, : self.dX]
            U[:, t] = xu[:, self.dX :]
        # Final state
        X[:, T] = rng.multivariate_normal(self.X_mean[T], self.X_covar[T], size=N)

        return X.reshape(size + (T + 1, self.dX)), U.reshape(size + (T, self.dU))  # Reshape to desired shape

import numpy as np


class TrajectoryDistribution:
    """A normally distributed trajectory."""

    def __init__(self, mean: np.ndarray, covar: np.ndarray, dX: int, dU: int = None) -> None:
        """Initialize this `TrajectoryDistribution`.

        Args:
            mean: (..., dX+dU) Means of the trajectory distribution
            covar: (..., dX+dU, dX+dU) Covariances of the trajectory distribution
            dX: Dimension of state space
            dU: Dimension of action space, may be inferred if not explicitly stated
        """
        if dU is None:
            dU = mean.shape[-1] - dX

        # Check shapes
        assert mean.shape[-1:] == (dX + dU, ), f"{mean.shape[-1:]} != {(dX + dU,)}"
        assert covar.shape[-2:] == (dX + dU, dX + dU), f"{covar.shape[-2:]} != {(dX + dU, dX + dU)}"
        assert mean.shape[:-1] == covar.shape[:-2], f"{mean.shape[:-1]} != {covar.shape[:-2]}"

        self.mean = mean
        self.covar = covar
        self.dX = dX
        self.dU = dU

    @property
    def X_mean(self):
        """Get state component of trajectory means."""
        return self.mean[..., :self.dX]

    @property
    def X_covar(self):
        """Get state component of trajectory covariances."""
        return self.covar[..., :self.dX, :self.dX]

    @property
    def U_mean(self):
        """Get action component of trajectory means."""
        return self.mean[..., self.dX:]

    @property
    def U_covar(self):
        """Get action component of trajectory covariances."""
        return self.covar[..., self.dX:, self.dX:]

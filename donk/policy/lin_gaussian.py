import numpy as np

from donk.models import TimeVaryingLinearGaussian
from donk.policy.policy import Policy


class LinearGaussianPolicy(Policy, TimeVaryingLinearGaussian):
    """Time-varying linear Gaussian policy.

    `p(u|t,x) ~ N(K_t * x + k_t, covar_t)`
    """

    def __init__(self, K: np.ndarray, k: np.ndarray, covar: np.ndarray = None, inv_covar: np.ndarray = None):
        """Initialize this LinearGaussianPolicy object.

        Must provide either covariance or precision, or both.

        Args:
            K: (T, dU, dX), Linear term
            k: (T, dU), Constant term
            covar: (T, dU, dU), Covariances
            inv_covar: (T, dU, dU), Inverse covariances, also called precision.
        """
        TimeVaryingLinearGaussian.__init__(self, K, k, covar, inv_covar)
        _, self.dU, self.dX = K.shape

    K = TimeVaryingLinearGaussian.coefficients
    k = TimeVaryingLinearGaussian.intercept

    def act(self, x: np.ndarray, t: int, noise: np.ndarray = None):
        """Decide an action for the given state at the current timestep.

        Uses the TVLG model to predict the next action.

        Args:
            x: (dX,) Current state
            t: Current timestep
            noise: (dU,) Action noise, may be `None` to sample without noise

        Returns:
            u: (dU,) Selected action
        """
        return self.predict(x, t, noise)

    def __str__(self) -> str:
        return f"LinearGaussianPolicy[T={self.T}, dX={self.dX}, dU={self.dU}]"

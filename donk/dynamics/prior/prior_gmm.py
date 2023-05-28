import numpy as np
from sklearn.mixture import GaussianMixture

from donk.dynamics.prior import DynamicsPrior, NormalInverseWishart


class GMMPrior(DynamicsPrior):
    """A Gaussian Mixture Model (GMM) based prior."""

    def __init__(self, n_clusters: int, random_state=None) -> None:
        """Initialize this `GMMPrior`."""
        self.gmm = GaussianMixture(n_components=n_clusters, random_state=random_state)

    def update(self, XUX: np.ndarray) -> None:
        """Update the prior.

        Args:
            XUX: (N, dX+dU+dX), transitions
        """
        self.N = XUX.shape[0]
        self.gmm.fit(XUX)

        # Enable warm start for further updates
        self.gmm.warm_start = True

    def eval(self, XUX: np.ndarray) -> NormalInverseWishart:
        """Evaluate the prior for the given transitions.

        Args:
            XUX: (N, dX+dU+dX), transitions
        """
        d = XUX.shape[1]
        wts = np.mean(self.gmm.predict_proba(XUX), axis=0)

        mu0 = np.sum(wts[:, np.newaxis] * self.gmm.means_, axis=0)
        Phi = np.sum(wts[:, np.newaxis, np.newaxis] * self.gmm.covariances_, axis=0)

        # Proportional number of samples in corresponding clusters
        N = np.sum(wts * self.gmm.weights_) * self.N

        return NormalInverseWishart.non_informative_prior(d).posterior(mu0, Phi, N)

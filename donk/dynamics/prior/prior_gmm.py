import numpy as np
from sklearn.mixture import GaussianMixture

from donk.dynamics.prior import DynamicsPrior


class GMMPrior(DynamicsPrior):

    def __init__(self, n_clusters, random_state=None) -> None:
        self.gmm = GaussianMixture(n_components=n_clusters, random_state=random_state)

    def update(self, XUX):
        """Updates the prior.

        Args:
            XUX: Transitions with shape (N, dX+dU+dX)
        """
        self.gmm.fit(XUX)

        # Enable warm start for further updates
        self.gmm.warm_start = True

    def eval(self, XUX):
        """Evaluate prior.

        Args:
            XUX: Transitions with shape (N, dX+dU+dX)

        Returns:
            mu0, Phi, m, n0
        """
        wts = np.mean(self.gmm.predict_proba(XUX), axis=0)

        mu0 = np.sum(wts[:, np.newaxis] * self.gmm.means_, axis=0)
        Phi = np.sum(wts[:, np.newaxis, np.newaxis] * self.gmm.covariances_, axis=0)

        # Use 1 for m and n0 instead of correct values as suggest by Fu et al. to compensate for discrepancies in
        # population size between prior and sampling.
        # see: https://arxiv.org/abs/1509.06841
        m = n0 = 1

        return mu0, Phi, m, n0

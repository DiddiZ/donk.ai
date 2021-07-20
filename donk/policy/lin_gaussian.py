from donk.policy import Policy
from donk.utils.batched import batched_inv_spd, batched_cholesky


class LinearGaussianPolicy(Policy):
    """Time-varying linear Gaussian policy.

    u ~ N(K_t * x + k_t, pol_covar_t)
    """

    def __init__(self, K, k, pol_covar=None, inv_pol_covar=None):
        """Initialize this LinearGaussianPolicy object.

        Args:
            K: (T, dU, dX), Linear term
            k: (T, dU), Constant term
            pol_covar: (T, dU, dU), Covariances
        """
        if pol_covar is None and inv_pol_covar is None:
            raise ValueError('Must provide pol_covar or inv_pol_covar.')

        Policy.__init__(self)
        self.T, self.dU, self.dX = K.shape

        # Check shapes
        assert K.shape == (self.T, self.dU, self.dX), f"{K.shape} != {(self.T, self.dU, self.dX )}"
        assert k.shape == (self.T, self.dU), f"{k.shape} != {(self.T, self.dU)}"
        if pol_covar is not None:
            assert pol_covar.shape == (self.T, self.dU, self.dU), f"{pol_covar.shape} != {(self.T, self.dU, self.dU)}"
        if inv_pol_covar is not None:
            assert inv_pol_covar.shape == (self.T, self.dU, self.dU), f"{inv_pol_covar.shape} != {(self.T, self.dU, self.dU)}"

        self.K = K
        self.k = k
        # Compute covariance from precision, if neccesary.
        self.pol_covar = pol_covar if pol_covar is not None else batched_inv_spd(batched_cholesky(inv_pol_covar))
        self.chol_pol_covar = batched_cholesky(self.pol_covar)
        # Compute precision from covariance, if neccesary.
        self.inv_pol_covar = inv_pol_covar if inv_pol_covar is not None else batched_inv_spd(self.chol_pol_covar)

    def act(self, x, t: int, noise=None):
        """Decides an action for the given state at the current timestep.

        Samples action from the Gaussian distributing given by u ~ N(K_t * x + k_t, pol_covar_t).

        Args:
            x: (dX,) Current state
            t: Current timestep, required
            noise: (dU,) Action noise

        Returns:
            u: (dU,) Selected action
        """
        u = self.K[t].dot(x) + self.k[t]
        if noise is not None:
            u += self.chol_pol_covar[t].dot(noise)
        return u

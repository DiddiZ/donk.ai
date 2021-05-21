import numpy as np
from donk.dynamics import DynamicsModel
from donk.utils.batched import batched_cholesky


class LinearDynamics(DynamicsModel):
    """Stochastic linear dynamics model."""

    def __init__(self, Fm, fv, dyn_covar):
        """Initialize this LinearDynamics object.

        Args:
            Fm: (T, dX, dX + dU), Linear term
            fv: (T, dX), Constant term
            dyn_covar: (T, dX, dX), Covariances

        """
        self.T, self.dX = fv.shape
        self.dU = Fm.shape[2] - self.dX

        # Check shapes
        assert Fm.shape == (self.T, self.dX, self.dX + self.dU), f"{Fm.shape} != {(self.T, self.dX, self.dX + self.dU)}"
        assert fv.shape == (self.T, self.dX), f"{fv.shape} != {(self.T, self.dX)}"
        assert dyn_covar.shape == (self.T, self.dX, self.dX), f"{dyn_covar.shape} != {(self.T, self.dX, self.dX)}"

        self.Fm = Fm
        self.fv = fv
        self.dyn_covar = dyn_covar
        self.chol_dyn_covar = None

    def predict(self, x, u, t: int, noise=None):
        """Predict the next state.

        Samples next state from the Gaussian distributing given by x' ~ N(Fm_t * [x; u] + fv_t, dyn_covar_t).

        Args:
            x: (dX,) Current state
            u: (dU,) Current action
            t: Current timestep
            noise: (dX,) State noise, may be `None` to sample without noise

        Returns:
            x: (dX,) Next state
        """
        next_x = self.Fm[t, :, :self.dX].dot(x) + self.Fm[t, :, self.dX:].dot(u) + self.fv[t]
        if noise is not None:  # Add noise
            if self.chol_dyn_covar is None:  # Compute Cholesky decompositions if required
                self.chol_dyn_covar = batched_cholesky(self.dyn_covar)
            next_x += self.chol_dyn_covar[t].dot(noise)
        return next_x


def fit_lr(X, U, prior=None, regularization=1e-6):
    """Fit dynamics with least squares linear regression.

    Args:
        X: (N, T, dX), States
        U: (N, T, dU), Actions
        regularization: Added to the diagonal of the joint distribution variance. Ensures matrix is not singular.
    """
    N, T, dX = X.shape
    _, _, dU = U.shape
    dXU = dX + dU

    if N == 1:
        raise ValueError("Cannot fit dynamics to 1 sample")

    Fm = np.empty([T - 1, dX, dX + dU])
    fv = np.empty([T - 1, dX])
    dyn_covar = np.empty([T - 1, dX, dX])

    sig_reg = regularization * np.eye(dXU)

    # Perform regression for all time steps
    for t in range(T - 1):
        xux = np.c_[X[:, t], U[:, t], X[:, t + 1]]
        empmu = np.mean(xux, axis=0)
        empsig = (xux - empmu).T.dot(xux - empmu) / N

        if prior is None:
            mu = empmu
            sigma = empsig
        else:
            mu0, Phi, m, n0 = prior.eval(xux)
            mu = empmu  # Instead of using the correct one, suggested by Finn et al. in gps repo
            # mu = (m * mu0 + n0 * empmu) / (m + n0) # The correct one
            sigma = (Phi + (N - 1) * empsig + (N * m) / (N + m) * (empmu - mu0).T.dot(empmu - mu0)) / (N + n0)

        # Apply regularization to ensure non-sigularity
        sigma[:dXU, :dXU] += sig_reg

        # Condition on x_t, u_t
        Fm[t] = np.linalg.solve(sigma[:dXU, :dXU], sigma[:dXU, dXU:]).T
        fv[t] = mu[dXU:] - Fm[t].dot(mu[:dXU])
        dyn_covar[t] = sigma[dXU:, dXU:] - Fm[t].dot(sigma[:dXU, :dXU]).dot(Fm[t].T)

        # Symmetrize
        dyn_covar[t] = (dyn_covar[t].T + dyn_covar[t]) / 2

    return Fm, fv, dyn_covar

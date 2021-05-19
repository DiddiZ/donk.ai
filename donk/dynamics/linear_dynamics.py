import numpy as np


class LinearDynamics:
    """Stochastic linear dynamics model."""

    def __init__(self, Fm, fv, dyn_covar):
        """Initialize this LinearDynamics object.

        Args:
            Fm: (T, dX, dX + dU), linear term
            fv: (T, dX), constant term
            dyn_covar: (T, dX, dX), covariances

        """
        self.T, self.dX = fv.shape
        self.dU = Fm.shape[2] - self.dX

        self.Fm = Fm
        self.fv = fv
        self.dyn_covar = dyn_covar


def fit_lr(X, U, regularization=1e-6):
    """Fit dynamics with least squares linear regression.

    Args:
        X: States, shape (N, T, dX)
        U: Actions, shape (N, T, dU)
        regularization: Added to the diagonal of the joint distribution variance. Ensures matrix is not singular.
    """
    N, T, dX = X.shape
    _, _, dU = U.shape
    dXU = dX + dU

    if N == 1:
        raise ValueError("Cannot fit dynamics on 1 sample")

    Fm = np.empty([T - 1, dX, dX + dU])
    fv = np.empty([T - 1, dX])
    dyn_covar = np.empty([T - 1, dX, dX])

    sig_reg = regularization * np.eye(dXU)

    # Perform regression for all time steps
    for t in range(T - 1):
        xux = np.c_[X[:, t], U[:, t], X[:, t + 1]]
        xux_mean = np.mean(xux, axis=0)
        empsig = (xux - xux_mean).T.dot(xux - xux_mean) / N

        sigma = empsig
        sigma[:dXU, :dXU] += sig_reg  # Apply regularization to ensure non-sigularity

        Fm[t] = np.linalg.solve(sigma[:dXU, :dXU], sigma[:dXU, dXU:]).T
        fv[t] = xux_mean[dXU:] - Fm[t].dot(xux_mean[:dXU])
        dyn_covar[t] = sigma[dXU:, dXU:] - Fm[t].dot(sigma[:dXU, :dXU]).dot(Fm[t].T)
    return Fm, fv, dyn_covar

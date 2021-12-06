from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import multivariate_normal

from donk.dynamics import DynamicsModel
from donk.dynamics.prior import DynamicsPrior
from donk.models import TimeVaryingLinearGaussian
from donk.utils.batched import regularize, symmetrize

multivariate_normal_logpdf = np.vectorize(multivariate_normal.logpdf, signature="(x),(x),(x,x)->()")


class LinearDynamics(DynamicsModel, TimeVaryingLinearGaussian):
    """Stochastic linear dynamics model."""

    def __init__(self, F: np.ndarray, f: np.ndarray, covar: np.ndarray):
        """Initialize this LinearDynamics object.

        Args:
            Fm: (T, dX, dX + dU), Linear term
            fv: (T, dX), Constant term
            dyn_covar: (T, dX, dX), Covariances

        """
        TimeVaryingLinearGaussian.__init__(self, F, f, covar)
        _, self.dX = f.shape
        self.dU = F.shape[2] - self.dX

    F = TimeVaryingLinearGaussian.coefficients
    f = TimeVaryingLinearGaussian.intercept

    def predict(self, x: np.ndarray, u: np.ndarray, t: int = None, noise: np.ndarray = None):
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
        return TimeVaryingLinearGaussian.predict(self, np.concatenate([x, u], axis=-1), t, noise)

    def __str__(self) -> str:
        return f"LinearDynamics[T={self.T}, dX={self.dX}, dU={self.dU}]"

    def log_prob(self, X: np.ndarray, U: np.ndarray) -> float:
        """Compute log probabilities for transitions to occur under this dynamics.

        Args:
            X: (..., T+1, dX) States
            U: (..., T, dU) Actions
        Returns:
            log_prob: (..., T) log transition probabilities
        """
        means = self.predict(X[..., :-1, :], U, t=None)
        return multivariate_normal_logpdf(X[..., 1:, :], means, self.covar)

    def evaluate(self, output_dir: str | Path, X_train: np.ndarray, U_train: np.ndarray, X_test: np.ndarray, U_test: np.ndarray):
        """Create diagnostics and evaluation plots for this dynamics model.

        Args:
            output_dir: Directory to write the plots to
            X_train: (N_train, T+1, dX), Train set states
            U_train: (N_train, T, dU), Train set actions
            X_test: (N_test, T+1, dX), Test set states
            U_test: (N_test, T, dU), Test set actions
        """
        import pandas as pd

        import donk.visualization as vis

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        X = np.concatenate([X_train, X_test], axis=0)
        U = np.concatenate([U_train, U_test], axis=0)

        # Compute prediction errors
        N_test = X_test.shape[0]
        prediction = np.empty((N_test, self.T, self.dX))
        for n in range(N_test):
            for t in range(self.T):
                prediction[n, t] = self.predict(X_test[n, t], U_test[n, t], t, noise=None)

        # Create plots
        if output_dir is not None:
            vis.visualize_linear_model(
                output_dir / "parameters.pdf",
                self.Fm,
                self.fv,
                self.dyn_covar,
                x=np.mean(np.c_[X_train[:, :-1], U_train], axis=0),  # Sample mean
            )
            vis.visualize_coefficients(str(output_dir / "coefficients_{:02d}.pdf"), self.Fm)
            vis.visualize_covariance(output_dir / "covariance.pdf", self.dyn_covar.mean(axis=0))
            vis.visualize_prediction(str(output_dir / "prediction_{:02d}.pdf"), prediction, X_test[:, 1:])
            vis.visualize_prediction_error(output_dir / "error.pdf", prediction, X_test[:, 1:])
            vis.visualize_predictor_target_correlation(
                output_dir / "state_correlation.pdf",
                X=np.concatenate([X[:, :-1], U], axis=-1).reshape(-1, self.dX + self.dU),
                Y=X[:, 1:].reshape(-1, self.dX),
                xlabel="$xu_t$",
                ylabel="$x_{t+1}$",
            )
            vis.visualize_predictor_target_scatter(
                output_dir / "state_correlation_scatter.png", X=np.concatenate([X[:, :-1], U], axis=-1), Y=X[:, 1:]
            )

        # Gather statistics
        statistics = pd.DataFrame(
            [
                ("T", self.T),
                ("dX", self.dX),
                ("dU", self.dU),
                ("coefficients_variance", np.var(self.Fm, axis=0).mean()),
                ("prediction_error", np.mean((prediction - X_test[:, 1:])**2)),
            ],
            columns=["metric", "score"],
        )
        if output_dir is not None:
            statistics.to_csv(output_dir / "statistics.csv", index=False)

        return statistics


def fit_lr(X: np.ndarray, U: np.ndarray, prior: DynamicsPrior = None, regularization=1e-6) -> LinearDynamics:
    """Fit dynamics with least squares linear regression.

    Args:
        X: (N, T+1, dX), States
        U: (N, T, dU), Actions
        prior: DynamicsPrior to be used. May be `None` to fit without prior.
        regularization: Eigenvalues of the joint distribution variance are lifted to this value. Ensures matrix is not singular.
    """
    N, _, dX = X.shape
    _, T, dU = U.shape
    dXU = dX + dU

    if N <= 1:
        raise ValueError(f"Cannot fit dynamics to {N} sample(s)")

    Fm = np.empty([T, dX, dX + dU])
    fv = np.empty([T, dX])
    dyn_covar = np.empty([T, dX, dX])

    # Perform regression for all time steps
    for t in range(T):
        xux = np.c_[X[:, t], U[:, t], X[:, t + 1]]
        empmu = np.mean(xux, axis=0)
        empsig = (xux - empmu).T @ (xux - empmu) / N

        if prior is None:
            mu = empmu
            sigma = empsig
        else:
            prior_dist = prior.eval(xux)
            posterior_dist = prior_dist.posterior(empmu, empsig, N)

            # Override the size of the prior
            # TODO More elegant solution
            posterior_dist.N_mean = 1
            posterior_dist.N_covar = -(dX + dU + dX)

            mu = empmu  # Instead of using the estimation, suggested by Finn et al. in https://github.com/cbfinn/gps
            sigma = posterior_dist.map_covar()

        # Apply regularization to ensure non-sigularity
        min_eigval = np.min(np.linalg.eigvalsh(sigma[:dXU, :dXU]))
        if min_eigval < regularization:
            # Lift eigenvalues lowest eigenvalue to regularization
            regularize(sigma[:dXU, :dXU], regularization - min_eigval)

        # Condition on x_t, u_t
        Fm[t] = np.linalg.solve(sigma[:dXU, :dXU], sigma[:dXU, dXU:]).T
        fv[t] = mu[dXU:] - Fm[t] @ (mu[:dXU])
        dyn_covar[t] = sigma[dXU:, dXU:] - Fm[t] @ sigma[:dXU, :dXU] @ Fm[t].T

    symmetrize(dyn_covar)
    return LinearDynamics(Fm, fv, dyn_covar)

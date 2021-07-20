import numpy as np
from donk.utils import symmetrize


def forward(dynamics, policy, X_0_mean, X_0_covar):
    """Perform LQR forward pass.

    Computes state-action marginals from dynamics and policy.

    Args:
        dynamics: A LinearGaussianPolicy.
        policy: A LinearDynamics.
        X_0_mean: (dX, ) Mean of initial state distribution.
        X_0_covar: (dX, dX) Covariance of initial state distribution.

    Returns:
        traj_mean: (T, dX+dU) mean state-action vectors.
        traj_covar: (T, dX+dU, dX+dU) state-action covariance matrices.

    """
    T, dX, dU = dynamics.T, dynamics.dX, dynamics.dU

    traj_mean = np.empty((T, dX + dU))
    traj_covar = np.empty((T, dX + dU, dX + dU))

    # Set initial state dist
    traj_mean[0, :dX] = X_0_mean
    traj_covar[0, :dX, :dX] = X_0_covar

    # For convenicence
    Fm, fv, dyn_covar = dynamics.Fm, dynamics.fv, dynamics.dyn_covar
    K, k, pol_covar = policy.K, policy.k, policy.pol_covar

    for t in range(T):
        # u_t = K_t*x_t + k_t
        traj_mean[t, dX:] = K[t] @ traj_mean[t, :dX] + k[t]
        # TODO Formulas
        traj_covar[t, dX:, :dX] = K[t] @ traj_covar[t, :dX, :dX]
        traj_covar[t, :dX, dX:] = traj_covar[t, dX:, :dX].T
        traj_covar[t, dX:, dX:] = traj_covar[t, dX:, :dX] @ K[t].T + pol_covar[t]

        if t < T - 1:
            # x_t+1 = Fm_t*[x_t;u_t] + fv_t
            traj_mean[t + 1, :dX] = Fm[t] @ traj_mean[t] + fv[t]
            # TODO Formula
            traj_covar[t + 1, :dX, :dX] = Fm[t] @ traj_covar[t] @ Fm[t].T + dyn_covar[t]

        symmetrize(traj_covar[t])
    return traj_mean, traj_covar

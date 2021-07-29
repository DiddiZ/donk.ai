import numpy as np
from scipy.linalg import solve

from donk.dynamics import LinearDynamics
from donk.policy import LinearGaussianPolicy
from donk.utils import regularize, symmetrize


def backward(dynamics: LinearDynamics, C, c, gamma=1) -> LinearGaussianPolicy:
    """Perform LQR backward pass.

    `C` is required to be symmetric.
    `C[dU:, dU:]` is required to be s.p.d.
    At `C[T]` only `C[T, :dX, :dX]` needs to be defined. The other costs at time `T+1` can be undefined.

    Args:
        dynamics: A LinearDynamics object.
        C: (T+1, dX+dU, dX+dU) Quadratic costs
        c: (T+1, dX+dU) Linear costs
        gamma: Discount factor for future rewards

    Returns:
        traj_distr: A new linear Gaussian policy.

    """
    T, dX, dU = dynamics.T, dynamics.dX, dynamics.dU

    K = np.empty((T, dU, dX))
    k = np.empty((T, dU))
    pol_covar = np.empty((T, dU, dU))
    inv_pol_covar = np.empty((T, dU, dU))

    # Set value of final state
    V = C[T, :dX, :dX]
    v = c[T, :dX]

    # For convenicence
    Fm, fv = dynamics.Fm, dynamics.fv

    # Compute state-action-state function at each time step.
    for t in reversed(range(T)):
        # Compute Q function.
        Q = C[t] + gamma * Fm[t].T @ V @ Fm[t]
        q = c[t] + gamma * Fm[t].T @ (V @ fv[t] + v)
        symmetrize(Q)

        # Compute inverse of Q function action component (as it's required explicitely anyway).
        Q_uu_inv = solve(Q[dX:, dX:], np.eye(dU), assume_a="pos")
        symmetrize(Q_uu_inv)

        # Compute controller parameters
        # K_t = -Q_{uut}^-1 * Q_{uxt}
        K[t] = -Q_uu_inv @ Q[dX:, :dX]
        # K_t = -Q_{uut}^-1 * q_{ut}
        k[t] = -Q_uu_inv @ q[dX:]
        # pol_covar_t = Q_{uut}^-1
        pol_covar[t] = Q_uu_inv
        inv_pol_covar[t] = Q[dX:, dX:]

        # Compute value function.
        V = Q[:dX, :dX] + Q[:dX, dX:] @ K[t]
        v = q[:dX] + Q[:dX, dX:] @ k[t]
        symmetrize(V)

    return LinearGaussianPolicy(K, k, pol_covar, inv_pol_covar)


def forward(dynamics: LinearDynamics, policy: LinearGaussianPolicy, X_0_mean, X_0_covar, regularization=1e-6):
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
        regularize(traj_covar[t], regularization)
    return traj_mean, traj_covar


def extended_costs_kl(prev_pol: LinearGaussianPolicy):
    """Compute expansion of extended cost used in the iLQR backward pass.

    The extended cost function is -log p(u_t | x_t) with p being the previous trajectory distribution.
    Thus, rewarding similarity of actions to the previous policy.

    Returns:
        C: Quadratic term of extended costs
        c: Linear term of extended costs
    """
    T, dX, dU = prev_pol.T, prev_pol.dX, prev_pol.dU

    C = np.empty((T, dX + dU, dX + dU))
    c = np.empty((T, dX + dU))

    # For convenicence
    K, k, inv_pol_covar = prev_pol.K, prev_pol.k, prev_pol.inv_pol_covar

    for t in range(T):
        C[t, :dX, :dX] = K[t].T @ inv_pol_covar[t] @ K[t]
        C[t, :dX, dX:] = -K[t].T @ inv_pol_covar[t]
        C[t, dX:, :dX] = -inv_pol_covar[t] @ K[t]
        C[t, dX:, dX:] = inv_pol_covar[t]
        c[t, :dX] = K[t].T @ inv_pol_covar[t] @ k[t]
        c[t, dX:] = -inv_pol_covar[t] @ k[t]

    return C, c


def kl_divergence_action(X, pol: LinearGaussianPolicy, prev_pol: LinearGaussianPolicy):
    """Compute KL divergence between new and previous trajectory distributions.

    Args:
        X: (T, dX), mean of states to compare actions for.
        pol: LinearGaussianPolicy, new policy.
        prev_pol: LinearGaussianPolicy, previous policy.

    Returns:
        kl_div: The mean KL divergence between the new and previous actions over time.

    See:
        https://web.stanford.edu/~jduchi/projects/general_notes.pdf Chapter 9
    """
    T, dU = pol.T, pol.dU

    kl_div = 0
    for t in range(T):
        delta_u = (prev_pol.K[t] - pol.K[t]) @ X[t] + prev_pol.k[t] - pol.k[t]
        kl_div += 0.5 * (
            # tr(sigma_1^-1 * sigma_0)
            np.einsum("ij,ij->", prev_pol.inv_pol_covar[t], pol.pol_covar[t]) +
            # (mu_1 - mu_0)^T * sigma_1^-1 * (mu_1 - mu_0)
            delta_u.T @ prev_pol.inv_pol_covar[t] @ delta_u +
            # -k
            -dU +
            # log(det(sigma_1) / det(sigma_0))
            2 * sum(np.log(np.diag(prev_pol.chol_pol_covar[t])) - np.log(np.diag(pol.chol_pol_covar[t])))
        )

    return kl_div / T

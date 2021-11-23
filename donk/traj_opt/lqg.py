from dataclasses import dataclass

import numpy as np
from scipy import optimize
from scipy.linalg import solve

from donk.costs import QuadraticCosts
from donk.dynamics import LinearDynamics
from donk.policy import LinearGaussianPolicy
from donk.samples import StateDistribution
from donk.traj_opt.traj_dist import TrajectoryDistribution
from donk.utils import regularize, symmetrize, trace_of_product


@dataclass
class ILQRStepResult:
    eta: float
    policy: LinearGaussianPolicy
    kl_div: float
    expected_costs: float
    trajectory: TrajectoryDistribution


class ILQR:
    """iLQG trajectory optimization."""

    def __init__(
        self,
        dynamics: LinearDynamics,
        prev_pol: LinearGaussianPolicy,
        costs: QuadraticCosts,
        initial_state: StateDistribution,
    ) -> None:
        """Initialize this object.

        Args:
            dynamics: Time-varying linear Gaussian dynamics model
            prev_pol: Policy to contrain the new policy to
            initial_state: initial state distribution mean
            costs: (T, dX+dU, dX+dU): Quadratic cost function
        """
        self.dynamics = dynamics
        self.prev_pol = prev_pol
        self.costs = costs
        self.initial_state = initial_state

        # Compute extended costs
        self.C_kl, self.c_kl = extended_costs_kl(prev_pol)

    def step(self, eta) -> ILQRStepResult:
        """Perfrom the unconstrained optimization under the given Lagrange multiplier.

        Args:
            eta: Lagrange multiplier
        """
        # Compute extended costs for the given Lagrangian multiplier
        C_ext = self.costs.C / eta
        C_ext[:-1] += self.C_kl
        c_ext = self.costs.c / eta
        c_ext[:-1] += self.c_kl

        # Perform unconstrained trajectory optimization under the extended costs
        pol = backward(self.dynamics, C_ext, c_ext)

        # Compute KL-divergence and expected costs of the new policy
        traj = forward(self.dynamics, pol, self.initial_state)
        kl_div = kl_divergence_action(traj.X_mean, pol, self.prev_pol)
        expected_costs = np.sum(self.costs.expected_costs(traj.mean, traj.covar))

        return ILQRStepResult(eta, pol, kl_div, expected_costs, traj)

    def sample_surface(self, min_eta: float = 1e-6, max_eta: float = 1e16, N: int = 100):
        """Sample the Lagrangian at different values for eta.

        For visualization/debugging purposes.

        Args:
            min_eta: Minimal value of the Lagrangian multiplier
            max_eta: Maximal value of the Lagrangian multiplier
            N: Number of samples. Samples are equally distanced in log space to base 10.
        """
        results = [self.step(eta) for eta in np.logspace(np.log10(min_eta), np.log10(max_eta), num=N)]
        return results

    def optimize(self, kl_step: float, min_eta: float = 1e-6, max_eta: float = 1e16, rtol: float = 1e-2, full_history: bool = False):
        """Perform iLQG trajectory optimization.

        Args:
            kl_step: KL divergence threshold to previous policy
            min_eta: Minimal value of the Lagrangian multiplier
            max_eta: Maximal value of the Lagrangian multiplier
            rtol: Tolerance of found solution to kl_step. Levine et al. propose a value of 0.1 in "Learning Neural Network Policies with
                  Guided Policy Search under Unknown Dynamics", chapter 3.1
            full_history: Whether to return ahistory of all optimization steps, for debug purposes

        Returns:
            result: A `ILQRStepResult` or
                    a list of `ILQRStepResult` if `full_history` is enabled (in order they were visited)
        """
        results = [self.step(min_eta)]
        if results[0].kl_div > kl_step:
            # Find the point where kl divergence equals the kl_step
            def constraint_violation(log_eta):
                results.append(self.step(np.exp(log_eta)))
                return results[-1].kl_div - kl_step

            # Search root of the constraint violation
            # Perform search in log-space, as this requires much fewer iterations
            optimize.brentq(constraint_violation, np.log(min_eta), np.log(max_eta), rtol=rtol)

        return results if full_history else results[-1]


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
    F, f = dynamics.F, dynamics.f

    # Compute state-action-state function at each time step.
    for t in reversed(range(T)):
        # Compute Q function.
        Q = C[t] + gamma * F[t].T @ V @ F[t]
        q = c[t] + gamma * F[t].T @ (V @ f[t] + v)
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


def forward(
    dynamics: LinearDynamics,
    policy: LinearGaussianPolicy,
    initial_state: StateDistribution,
    regularization=1e-6,
) -> TrajectoryDistribution:
    """Perform LQR forward pass.

    Computes state-action marginals from dynamics and policy.

    Distribution of last time step only contains the state distribution, as there is no corresponding action.

    Args:
        dynamics: A LinearGaussianPolicy.
        policy: A LinearDynamics.
        X_0_mean: (dX, ) Mean of initial state distribution.
        X_0_covar: (dX, dX) Covariance of initial state distribution.

    Returns:
        traj_mean: (T+1, dX+dU) mean state-action vectors.
        traj_covar: (T+1, dX+dU, dX+dU) state-action covariance matrices.

    """
    T, dX, dU = dynamics.T, dynamics.dX, dynamics.dU

    traj_mean = np.empty((T + 1, dX + dU))
    traj_covar = np.empty((T + 1, dX + dU, dX + dU))

    # Set initial state dist
    traj_mean[0, :dX] = initial_state.mean
    traj_covar[0, :dX, :dX] = initial_state.covar

    # Set action part for final state
    traj_mean[T, dX:] = 0
    traj_covar[T] = 0

    # For convenicence
    F, f, dyn_covar = dynamics.F, dynamics.f, dynamics.covar
    K, k, pol_covar = policy.K, policy.k, policy.covar

    for t in range(T):
        # u_t = K_t*x_t + k_t
        traj_mean[t, dX:] = K[t] @ traj_mean[t, :dX] + k[t]
        # TODO Formulas
        traj_covar[t, dX:, :dX] = K[t] @ traj_covar[t, :dX, :dX]
        traj_covar[t, :dX, dX:] = traj_covar[t, dX:, :dX].T
        traj_covar[t, dX:, dX:] = traj_covar[t, dX:, :dX] @ K[t].T + pol_covar[t]

        # x_t+1 = Fm_t*[x_t;u_t] + fv_t
        traj_mean[t + 1, :dX] = F[t] @ traj_mean[t] + f[t]
        # TODO Formula
        traj_covar[t + 1, :dX, :dX] = F[t] @ traj_covar[t] @ F[t].T + dyn_covar[t]

        symmetrize(traj_covar[t])
        regularize(traj_covar[t], regularization)

    # Symmetrize state distribution of final state
    symmetrize(traj_covar[T, :dX, :dX])

    return TrajectoryDistribution(traj_mean, traj_covar, dX, dU)


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
    K, k, inv_pol_covar = prev_pol.K, prev_pol.k, prev_pol.inv_covar

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
            trace_of_product(prev_pol.inv_covar[t], pol.covar[t]) +
            # (mu_1 - mu_0)^T * sigma_1^-1 * (mu_1 - mu_0)
            delta_u.T @ prev_pol.inv_covar[t] @ delta_u +
            # -k
            -dU +
            # log(det(sigma_1) / det(sigma_0))
            2 * sum(np.log(np.diag(prev_pol.chol_covar[t])) - np.log(np.diag(pol.chol_covar[t])))
        )

    return kl_div / T

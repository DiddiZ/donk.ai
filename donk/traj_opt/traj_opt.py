import numpy as np

from donk.costs import CostFunction, QuadraticCosts
from donk.dynamics import LinearDynamics, linear_dynamics
from donk.dynamics.prior import DynamicsPrior
from donk.policy import LinearGaussianPolicy
from donk.samples import StateDistribution
from donk.traj_opt import ILQR, lqg


class TrajOptAlgorithm:
    """Algorithm for iterative LQR trajectory optimization."""

    def __init__(self, kl_step: float) -> None:
        self.kl_step = kl_step
        self.kl_step_mult: float = 1
        self.pol: LinearGaussianPolicy = None
        self.dyn: LinearDynamics = None
        self.x0: StateDistribution = None
        self.costs: QuadraticCosts = None

    def iteration(
        self,
        X: np.ndarray,
        U: np.ndarray,
        cost_function: CostFunction,
        prev_pol: LinearGaussianPolicy = None,
        prior: DynamicsPrior = None
    ):
        """Perform one iteration of trajectory optimization.

        Args:
            X: (N, T+1, dX) Train states
            U: (N, T, dU) Train actions
            cost_function: Cost function to minimize
            prev_pol: Policy to constrain trajectory optimization to
            prior: Optional prior for dynamics fit
        """
        prev_dyn = self.dyn
        prev_x0 = self.x0
        prev_costs = self.costs
        prev_pol = prev_pol if prev_pol is not None else self.pol

        # Fit initial state distribution
        self.x0 = StateDistribution.fit(X[:, 0])

        # Fit dynamics
        self.dyn = linear_dynamics.fit_lr(X, U, prior=prior, regularization=0)

        # Evaluate costs
        # Assume trajectory mean as operating point
        self.costs = cost_function.quadratic_approximation(np.mean(X, axis=0), np.mean(U, axis=0))

        # Perform trajectory optimization
        ilqr = ILQR(self.dyn, prev_pol, self.costs, self.x0)
        self.pol = ilqr.optimize(self.kl_step * self.kl_step_mult).policy

        # Update kl_step
        if prev_dyn is not None:
            self.kl_step_mult = lqg.step_adjust(
                self.kl_step_mult, self.dyn, self.pol, self.x0, self.costs, prev_dyn, prev_pol, prev_x0, prev_costs
            )

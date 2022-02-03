import numpy as np

import donk.datalogging as datalogging
from donk.costs import CostFunction, QuadraticCosts
from donk.dynamics import LinearDynamics, linear_dynamics
from donk.dynamics.prior import DynamicsPrior
from donk.policy import LinearGaussianPolicy
from donk.samples import StateDistribution
from donk.traj_opt import ILQR, lqg


class TrajOptAlgorithm:
    """Algorithm for iterative LQR trajectory optimization."""

    def __init__(
        self, kl_step: float, max_step_mult: float = 10, min_step_mult: float = 0.1, dynamics_regularization: float = 1e-6
    ) -> None:
        self.kl_step = kl_step
        self.kl_step_mult: float = 1
        self.max_step_mult: float = max_step_mult
        self.min_step_mult: float = min_step_mult
        self.dynamics_regularization: float = dynamics_regularization

        self.pol: LinearGaussianPolicy = None
        self.dyn: LinearDynamics = None
        self.x0: StateDistribution = None
        self.costs: QuadraticCosts = None
        self.ilqr: ILQR = None

    def iteration(
        self,
        X: np.ndarray,
        U: np.ndarray,
        cost_function: CostFunction,
        prev_pol: LinearGaussianPolicy = None,
        prior: DynamicsPrior = None,
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
        self.dyn = linear_dynamics.fit_lr(X, U, prior=prior, regularization=self.dynamics_regularization)

        # Evaluate costs
        # Assume trajectory mean as operating point
        self.costs = cost_function.quadratic_approximation(np.mean(X, axis=0), np.mean(U, axis=0))

        # Perform trajectory optimization
        ilqr = ILQR(self.dyn, prev_pol, self.costs, self.x0)
        self.pol = ilqr.optimize(self.kl_step * self.kl_step_mult).policy

        # Log
        datalogging.log(
            prev_pol=prev_pol,
            pol=self.pol,
            dyn=self.dyn,
            x0=self.x0,
            costs=self.costs,
            kl_step=self.kl_step * self.kl_step_mult,
        )

        # Update kl_step
        if prev_dyn is not None:
            with datalogging.Context("step_adjust"):
                self.kl_step_mult = lqg.step_adjust(
                    self.kl_step_mult, self.dyn, self.pol, self.x0, self.costs, prev_dyn, prev_pol, prev_x0, prev_costs, self.max_step_mult,
                    self.min_step_mult
                )

from abc import ABC, abstractmethod

import numpy as np

from donk.costs.quadratic_costs import QuadraticCosts


class CostFunction(ABC):
    """Base class for cost functions.

    Defines a `quadratic_approximation` method.
    """

    @abstractmethod
    def quadratic_approximation(self, X: np.ndarray, U: np.ndarray) -> QuadraticCosts:
        """Compute a quadratic approximation (2nd order Taylor) at the given trajectory.

        Args:
            X: (T+1, dX), states
            U: (T, dX), actions
        """

    @abstractmethod
    def compute_costs(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Evaluate costs for trajectories.

        Args:
            X: (..., T+1, dX), states
            U: (..., T, dU), actions

        Returns:
            costs: (..., T+1), Costs at each time step
        """

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class NormalInverseWishart:
    """A normal-inverse Wishart distribution."""
    mu0: np.ndarray
    Phi: np.ndarray
    m: float
    n0: float

    def estimate_mean(self, emp_mean: np.ndarray) -> np.ndarray:
        """Estimate posterior mean.

        Args:
            emp_mean: Emperical sample mean

        See:
            Fu et al. "One-Shot Learning of Manipulation Skills with Online Dynamics Adaptation and Neural Network Priors", eq. (1)
        """
        # TODO Verify formula is correct. Should n0 be N?
        return (self.m * self.mu0 + self.n0 * emp_mean) / (self.m + self.n0)

    def estimate_covar(self, emp_mean, emp_covar, N):
        """Estimate posterior covariance.

        Args:
            emp_mean: Emperical sample mean
            emp_covar: Emperical sample covariance
            N: Number of samples

        See:
            Fu et al. "One-Shot Learning of Manipulation Skills with Online Dynamics Adaptation and Neural Network Priors", eq. (1)
        """
        # TODO Verify formula is correct
        return (self.Phi + N * emp_covar + (N * self.m) / (N + self.m) * (emp_mean - self.mu0).T @ (emp_mean - self.mu0)) / (N + self.n0)


class DynamicsPrior(ABC):
    """A prior distribution for linear dynamics.

    These priors are given as normal-inverse-Wishart distributions.
    """

    @abstractmethod
    def update(self, XUX: np.ndarray) -> None:
        """Update the prior.

        Args:
            XUX: (N, dX+dU+dX), transitions
        """

    @abstractmethod
    def eval(self, XUX: np.ndarray) -> NormalInverseWishart:
        """Evaluate the prior for the given transitions.

        Args:
            XUX: (N, dX+dU+dX), transitions
        """

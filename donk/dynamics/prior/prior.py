from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class NormalInverseWishart:
    """A normal-inverse Wishart distribution."""

    mu0: np.ndarray
    Phi: np.ndarray
    N_mean: float
    N_covar: float

    def posterior(self, emp_mean: np.ndarray, emp_covar: np.ndarray, N: float):
        """Compute prior distibution after observing samples with the given sample distribution.

        Args:
            emp_mean: (d,) Sample mean
            emp_covar: (d, d) Sample covariance
            N: Number of samples. May be fractional

        See:
            Gelman et al. Bayesian Data Analysis, chapter 3.6
        """
        return NormalInverseWishart(
            mu0=(self.N_mean * self.mu0 + N * emp_mean) / (self.N_mean + N),
            Phi=self.Phi
            + N * emp_covar
            + self.N_mean * N / (self.N_mean + N) * np.einsum("i,j->ij", emp_mean - self.mu0, emp_mean - self.mu0),
            N_mean=self.N_mean + N,
            N_covar=self.N_covar + N,
        )

    def map_mean(self) -> np.ndarray:
        """Maximum aposteriory estimate for the mean."""
        return self.mu0.copy()

    def map_covar(self) -> np.ndarray:
        """Maximum aposteriory estimate for the covariance."""
        d = self.mu0.shape[-1]
        return self.Phi / (self.N_covar + d + 1)

    @staticmethod
    def non_informative_prior(d):
        """Create a non-onformative prior distribution.

        Args:
            d: Dimension
        """
        return NormalInverseWishart(
            mu0=np.zeros((d,)),
            Phi=np.zeros((d, d)),
            N_mean=0,
            N_covar=-(1 + d),
        )


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

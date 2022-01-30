from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from tqdm import tqdm

import donk.policy


class Sampler(ABC):

    @abstractmethod
    def take_sample(self, pol: donk.policy.Policy, T: int, condition, rng: np.random.Generator = None) -> tuple[np.ndarray, np.ndarray]:
        """Take one policy sample.

        Args:
            pol: Policy to sample
            T: Number of timesteps to sample
            condition: Initial condition to reset the environemnt into
            rng: Generator for action noise. `None` indicates no action noise.

        Returns:
            X: (T+1, dX) states
            U: (T, dU) states
        """

    def take_samples(self,
                     N: int,
                     pol: donk.policy.Policy,
                     T: int,
                     condition,
                     rng: np.random.Generator = None,
                     silent: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Takes multiple policy sample.

        Args:
            N: number of samples to take
            pol: Policy to sample
            T: Number of timesteps to sample
            condition: Initial condition to reset the environemnt into
            rng: Generator for action noise. `None` indicates no action noise.
            silent: Whether to show a tqdm progress bar

        Returns:
            X: (N, T+1, dX) states
            U: (N, T, dU) states
        """
        # Allocate space
        X = np.empty((N, pol.T + 1, pol.dX))
        U = np.empty((N, pol.T, pol.dU))

        # Take N samples
        for i in tqdm(range(N), desc="Sampling", disable=silent):
            X[i], U[i] = self.take_sample(pol, T, condition, rng)

        return X, U

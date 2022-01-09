from __future__ import annotations

from abc import abstractmethod

import numpy as np

from donk.policy import Policy, smooth_noise
from donk.samples.sampler import Sampler


class GymSampler(Sampler):

    def __init__(self, env, smooth_kernel: float = 0.0) -> None:
        super().__init__()

        self.env = env
        self.smooth_kernel = smooth_kernel

    @abstractmethod
    def convert_observation(self, obs) -> np.ndarray:
        """Convert one observation from the Gym environment to a flat numpy array."""

    def take_sample(self, pol: Policy, condition: int, rng: np.random.Generator = None) -> tuple[np.ndarray, np.ndarray]:
        """Take one policy sample.

        Args:
            pol: Policy to sample
            condition: Random seed for seeding the environment
            rng: Generator for action noise. `None` indicates no action noise.

        Returns:
            X: (T+1, dX) states
            U: (T, dU) states
        """
        T, dX, dU = pol.T, pol.dX, pol.dU

        # Set initial condition
        self.env.seed(condition)

        # Allocate space
        X = np.empty((T + 1, dX))
        U = np.empty((T, dU))

        # Observe initial state
        obs = self.env.reset()
        X[0] = self.convert_observation(obs)

        # Generate action noise
        if rng is not None:
            noise = rng.standard_normal((T, dU))
            if self.smooth_kernel > 0:  # Apply smoothing
                noise = smooth_noise(noise, self.smooth_kernel)
        else:  # No action noise
            noise = np.zeros((T, dU))

        # Perform policy rollout
        for t in range(T):
            U[t] = pol.act(X[t], t, noise[t])
            obs, _, done, _ = self.env.step(U[t])
            if done and t < T - 1:
                raise Exception(f'Iteration ended prematurely {t+1}/{T}')
            X[t + 1] = self.convert_observation(obs)

        return X, U

from __future__ import annotations

from abc import abstractmethod

import numpy as np

from donk.policy import Policy, smooth_noise
from donk.samples.sampler import Sampler


class GymSampler(Sampler):
    """Sampler for Gym environments."""

    def __init__(self, env, smooth_kernel: float = 0.0) -> None:
        """Initialize Gym sampler.

        Args:
            env: Gym environment
            smooth_kernel: Standard deviation of Gaussian kernel for smoothing actions. `0.0` indicates no smoothing.
        """
        super().__init__()

        self.env = env
        self.smooth_kernel = smooth_kernel

    @abstractmethod
    def convert_observation(self, obs) -> np.ndarray:
        """Convert one observation from the Gym environment to a flat numpy array."""

    def convert_action(self, u: np.ndarray) -> np.ndarray:
        """Convert one action from a flat numpy array to one action for the from the Gym environment."""
        return u

    def take_sample(
        self, pol: Policy, T: int, condition: int, rng: np.random.Generator = None
    ) -> tuple[np.ndarray, np.ndarray]:
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
        dX, dU = pol.dX, pol.dU

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

        # Perform policy rollout
        for t in range(T):
            U[t] = pol.act(X[t], t, noise[t] if rng is not None else None)
            obs, _, done, _ = self.env.step(self.convert_action(U[t]))
            if done and t < T - 1:
                raise Exception(f"Iteration ended prematurely {t+1}/{T}")
            X[t + 1] = self.convert_observation(obs)

        return X, U

from abc import ABC, abstractmethod

import numpy as np


class Policy(ABC):
    """Base class for policies.

    Defines an `act` method.
    """

    @abstractmethod
    def act(self, x: np.ndarray, t: int = None, noise: np.ndarray = None) -> np.ndarray:
        """Decide an action for the given state at the current timestep.

        For time-varying policies `t` is required.

        Args:
            x: (dX,) Current state
            t: Current timestep, may be `None`
            noise: (dU,) Action noise, may be `None` to sample without noise

        Returns:
            u: (dU,) Selected action
        """

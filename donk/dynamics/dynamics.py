from abc import ABC, abstractmethod

import numpy as np


class DynamicsModel(ABC):
    """Baseclass for dynamics models.

    Defines an `predict` method.
    """

    @abstractmethod
    def predict(self, x: np.ndarray, u: np.ndarray, t: int = None, noise: np.ndarray = None):
        """Predict the next state.

        For time-varying models `t` is required.

        Args:
            x: (dX,) Current state
            u: (dU,) Current action
            t: Current timestep, may be `None`
            noise: (dX,) State noise, may be `None` to sample without noise

        Returns:
            x: (dX,) Next state
        """

from abc import ABC, abstractmethod


class Policy(ABC):
    """Baseclass for policies.

    Defines an `act` method.
    """

    @abstractmethod
    def act(self, x, t: int = None, noise=None):
        """Decides an action for the given state at the current timestep.

        For time-varying policies `t` is required.

        Args:
            x: (dX,) Current state
            t: Current timestep, may be `None`
            noise: (dU,) Action noise, may be `None` to sample without noise

        Returns:
            u: (dU,) Selected action
        """
        pass

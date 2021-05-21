from abc import ABC, abstractmethod


class Policy(ABC):
    """Baseclass for policies.

    Defines an `act` method.
    """

    @abstractmethod
    def act(self, x, t: int = None, noise=None):
        """Decides an action for the given state at the current timestep.

        Args:
            x: (dX,) Current state
            t: Current timestep, may be `None`
            noise: (dU,) Action noise

        Returns:
            u: (dU,) Selected action
        """
        pass

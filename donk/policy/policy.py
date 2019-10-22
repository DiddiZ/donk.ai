from abc import ABC, abstractmethod


class Policy(ABC):
    """Baseclass for policies.

    Defines an `act` method.

    """

    @abstractmethod
    def act(self, x, t):
        """Decides an action for the given state at the current timestep.

        `t` can be used

        Args:
            x: State vector.
            t: Current timestep, may be `None`.

        Returns:
            A dU dimensional action vector.

        """
        pass

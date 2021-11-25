from abc import ABC, abstractmethod


class DynamicsPrior(ABC):

    @abstractmethod
    def update(self, XUX):
        """Updates the prior.

        Args:
            XUX: Transitions with shape (N, dX+dU+dX)
        """

    @abstractmethod
    def eval(self, XUX):
        """Evaluate prior.

        Args:
            XUX: Transitions with shape (N, dX+dU+dX)

        Returns:
            mu0, Phi, m, n0
        """

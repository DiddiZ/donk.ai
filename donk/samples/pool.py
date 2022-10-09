import numpy as np

from donk.dynamics import to_transitions


class TransitionPool:
    """Stores transition tuples."""

    def __init__(self, capacity: int = None) -> None:
        """Init this TransitionPool.

        Args:
            capacity: Site limit of this pool. Oldest transitions will be discarted first. `None` means no limit.
        """
        self._XUX = None
        self.capacity = capacity

    def add(self, X: np.ndarray, U: np.ndarray) -> None:
        """Add state-action sequences to the pool.

        Args:
            X: (N, T+1, dX), States
            U: (N, T, dU), Actions
        """
        # Check shapes
        assert X.shape[0] == U.shape[0], f"{X.shape[0]} != {U.shape[0]}"
        assert X.shape[1] == U.shape[1] + 1, f"{X.shape[1]} != {U.shape[1]+1}"
        if self._XUX is None:
            # Pool was empty, assume dimensions
            self.dX, self.dU = X.shape[-1], U.shape[-1]

            # Init pool
            self._XUX = to_transitions(X, U)
        else:
            # Sor safety, compare shapes before adding
            assert X.shape[-1] == self.dX, f"{X.shape[-1]} != {self.dX}"
            assert U.shape[-1] == self.dU, f"{U.shape[-1]} != {self.dU}"

            # Add to pool
            self._XUX = np.concatenate([self._XUX, to_transitions(X, U)], axis=0)
        if self.capacity is not None:
            self._XUX = self._XUX[-self.capacity :]

    def get_transitions(self, N: int = None):
        """Get transitions from the pool.

        Args:
            N: (optional) Return only the `N` newest transitions. Default is to return all.
        """
        XUX = self._XUX if N is None else self._XUX[-N:]

        # Create write-protected view
        XUX = XUX.view()
        XUX.flags.writeable = False

        return XUX

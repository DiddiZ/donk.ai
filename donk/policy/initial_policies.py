import numpy as np
from donk.policy.lin_gaussian import LinearGaussianPolicy


def constant_policy(T: int, dX: int, u, variance) -> LinearGaussianPolicy:
    """Generate an initial policy of constant actions with added noise.

    Args:
        T: Number of time steps
        dX: Size of state space
        u: (dU, ) Constant action to perform every step
        variance: (dU, ) Diagonal action variance
    """
    u, variance = np.asarray(u), np.asarray(variance)
    dU = u.shape[0]

    return LinearGaussianPolicy(
        K=np.zeros((T, dU, dX)),
        k=np.tile(u, (T, 1)),
        pol_covar=np.tile(np.diag(variance), (T, 1, 1)),
        inv_pol_covar=np.tile(np.diag(1 / variance), (T, 1, 1)),
    )

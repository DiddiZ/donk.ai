import numpy as np


def loss_l2(d, wp):
    """Evaluate and compute derivatives for l2 norm penalty.

    loss = 0.5 * d^2 * wp

    Args:
        d: (T, D) states to evaluate norm on.
        wp: (T, D) matrix with weights for each dimension and time step.

    Returns:
        l: (T,) cost at each timestep.
        lx: (T, D) first order derivative.
        lxx: (T, D, D) second order derivative.

    """
    # Get trajectory length.
    T, D = d.shape

    # Total cost
    # l = 0.5 * d^2 * wp
    l = 0.5 * np.sum(d**2 * wp, axis=1)

    # First order derivative
    # lx = d * wp
    lx = d * wp

    # Second order derivative
    # lxx = w
    lxx = np.einsum('ij,jk->ijk', wp, np.eye(D))

    return l, lx, lxx


def loss_log_cosh(d, wp):
    """Evaluate and compute derivatives for log-cosh loss.

    loss = log(cosh(d)) * wp

    Args:
        d: (T, D) states to evaluate norm on.
        wp: (T, D) matrix with weights for each dimension and time step.

    Returns:
        l: (T,) cost at each timestep.
        lx: (T, D) first order derivative.
        lxx: (T, D, D) second order derivative.

    """
    # Get trajectory length.
    T, D = d.shape

    # Total cost
    # l = log(cosh(d)) * wp
    l = np.sum(np.log(np.cosh(d)) * wp, axis=1)

    # First order derivative
    # lx = tanh(d) * wp
    lx = np.tanh(d) * wp

    # Second order derivative
    # lxx = wp / cosh^2(d)
    lxx = np.einsum('ij,jk->ijk', wp / np.cosh(d)**2, np.eye(D))

    return l, lx, lxx

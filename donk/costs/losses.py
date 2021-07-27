import numpy as np


def loss_combined(x, losses):
    """Evaluates and sums up multiple loss functions.

    Args:
        x: (T, dX) states, actual values.
        losses: List of tuples (`loss`, `kwargs`)
            `loss`: loss function to evaluate.
            `kwargs`: Addional arguments passed to the loss function (optional).

    """
    if len(losses) < 1:
        raise ValueError("loss_combined requred at least one loss function to sum up.")

    l, lx, lxx = losses[0][0](x, **losses[0][1])
    for loss in losses[1:]:
        l_, lx_, lxx_ = loss[0](x, **loss[1])
        l += l_
        lx += lx_
        lxx += lxx_

    return l, lx, lxx


def loss_l2(x, t, w):
    """Evaluate and compute derivatives for l2 norm penalty.

    loss = sum(0.5 * (x - t)^2 * w)

    Args:
        x: (T, dX) states, actual values.
        t: (T, dX) targets, expected values.
        w: (T, dX) weights, scale error of each feature at each timestep.

    Returns:
        l: (T,) cost at each timestep.
        lx: (T, D) first order derivative.
        lxx: (T, D, D) second order derivative.

    """
    # Get trajectory length.
    _, dX = x.shape

    d = x - t  # Error

    # Total cost
    # l = sum(0.5 * (x - t)^2 * w)
    l = 0.5 * np.sum(d**2 * w, axis=1)

    # First order derivative
    # lx = (x - t) * w
    lx = d * w

    # Second order derivative
    # lxx = w
    lxx = np.einsum('ij,jk->ijk', w, np.eye(dX))

    return l, lx, lxx


def loss_l1(x, t, w, alpha):
    """Evaluate and compute derivatives for l2 norm penalty.

    loss = sum(sqrt((x - t)^2 + alpha) * w)

    Args:
        x: (T, dX) states, actual values.
        t: (T, dX) targets, expected values.
        w: (T, dX) weights, scale error of each feature at each timestep.

    Returns:
        l: (T,) cost at each timestep.
        lx: (T, D) first order derivative.
        lxx: (T, D, D) second order derivative.

    """
    # Get trajectory length.
    _, dX = x.shape

    d = x - t  # Error
    abs_d = np.sqrt(alpha + d**2)

    # Total cost
    # l = sum(w * sqrt((x - t)^2 + alpha))
    l = np.sum(w * abs_d, axis=1)

    # First order derivative
    # lx = w * (x - t) / sqrt((x - t)^2 + alpha) * w
    lx = w * (x - t) / abs_d

    # Second order derivative
    # lxx = w * alpha / (((x-t)^2 + alpha)^(3/2))
    lxx = np.einsum('ij,jk->ijk', w * alpha / abs_d**3, np.eye(dX))

    return l, lx, lxx


def loss_log_cosh(x, t, w):
    """Evaluate and compute derivatives for log-cosh loss.

    loss = sum(log(cosh(x - t)) * w)

    Args:
        x: (T, dX) states, actual values.
        t: (T, dX) targets, expected values.
        w: (T, dX) weights, scale error of each feature at each timestep.

    Returns:
        l: (T,) cost at each timestep.
        lx: (T, D) first order derivative.
        lxx: (T, D, D) second order derivative.

    """
    # Get trajectory length.
    _, dX = x.shape

    d = x - t  # Error

    # Total cost
    # l = sum(log(cosh(x - t)) * w)
    l = np.sum(np.log(np.cosh(d)) * w, axis=1)

    # First order derivative
    # lx = tanh(x - t) * ws
    lx = np.tanh(d) * w

    # Second order derivative
    # lxx = w / cosh^2(x - t)
    lxx = np.einsum('ij,jk->ijk', w / np.cosh(d)**2, np.eye(dX))

    return l, lx, lxx

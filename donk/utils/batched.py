"""Batched versions of commin math operations."""
import numpy as np
from scipy.linalg import solve_triangular


def batched_inv_spd(a_chol):
    """Computes inverse of a batch of s.p.d. matrices from their cholesky decomposition.

    Exploits s.p.d.-ness for faster inverse.

    Args:
        a_chol: (T, dX, dX), batch of cholesky decompositions.

    """
    T, dX, _ = a_chol.shape
    a_inv = np.empty_like(a_chol)

    for t in range(T):
        a_inv[t] = solve_triangular(
            a_chol[t].T,
            solve_triangular(
                a_chol[t],
                np.eye(dX),
                lower=True,
                overwrite_b=True,
                check_finite=False,
            ),
            overwrite_b=True,
            check_finite=False,
        )
    return a_inv


def batched_cholesky(a):
    """Computes cholesky decomposition of a batch of s.p.d. matrices.

    Args:
        a: (T, dX, dX), batch of s.p.d. matrices

    Returns:
        Batch of lower triangular matrices.

    """
    T, dX, _ = a.shape
    a_chol = np.empty_like(a)

    for t in range(T):
        a_chol[t] = np.linalg.cholesky(a[t])
    return a_chol

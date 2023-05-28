"""Batched versions of commin math operations."""
import numpy as np
from scipy.linalg import solve_triangular


def batched_inv_spd(a_chol: np.ndarray) -> np.ndarray:
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


def batched_cholesky(a: np.ndarray) -> np.ndarray:
    """Computes cholesky decomposition of a batch of s.p.d. matrices.

    Args:
        a: (T, dX, dX), batch of s.p.d. matrices

    Returns:
        Batch of lower triangular matrices.

    """
    T, _, _ = a.shape
    a_chol = np.empty_like(a)

    for t in range(T):
        a_chol[t] = np.linalg.cholesky(a[t])
    return a_chol


def symmetrize(A: np.ndarray) -> np.ndarray:
    """Symmetrizes a matrix or a batch of matrices to eliminate numerical errors.

    Modifies the given matrix in-place.
    """
    A += np.swapaxes(A, -1, -2)
    A /= 2
    return A


def regularize(A: np.ndarray, regularization: float) -> np.ndarray:
    """Regularizes a matrix or a batch of matrices by adding a constant to the diagonal.

    Modifies the given matrix in-place.
    """
    idx = np.arange(A.shape[-1])
    A[..., idx, idx] += regularization
    return A


def trace_of_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute the trace of a matrix product Tr(A @ B).

    This is an O(n²) operation, as it avoids to compute the actial matrix product.

    Args:
        A: (..., n, m) Matrix, or stack of matrices
        B: (..., m, n) Matrix, or stack of matrices

    Returns:
        trace: Trace of stack of traces
    """
    return np.einsum("...ij,...ji->...", A, B)


def batched_multivariate_normal(mean: np.ndarray, covar: np.ndarray, N: int, rng: np.random.Generator) -> np.ndarray:
    """Draw `N` samples from `T` multivariate normal distributions each.

    Args:
        mean: (T, dX)
        covar: (T, dX, dX)
        N: Number of samples
        rng: Random number generator

    Returns:
        X: (N, T, dX) Radnom samples
    """
    T, dX = mean.shape

    X = np.empty((N, T, dX))
    for t in range(T):
        X[:, t] = rng.multivariate_normal(mean[t], covar[t], N)

    return X

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal


def test_batched_inv_spd() -> None:
    """Test batched_inv_spd implementation agains reference implementation using random values."""
    from donk.testing import random_spd
    from donk.utils.batched import batched_cholesky, batched_inv_spd

    T, dX = 10, 10
    rng = np.random.default_rng(0)

    a = random_spd((T, dX, dX), rng)
    a_chol_ref = np.empty_like(a)
    a_inv_ref = np.empty_like(a)
    for t in range(T):
        a_chol_ref[t] = np.linalg.cholesky(a[t])
        a_inv_ref[t] = np.linalg.inv(a[t])

    a_chol = batched_cholesky(a)
    a_inv = batched_inv_spd(a_chol)

    assert_allclose(a_chol_ref, a_chol)
    assert_allclose(a_inv_ref, a_inv)


def test_symmetrize() -> None:
    from donk.utils import symmetrize

    rng = np.random.default_rng(0)
    A = rng.normal(size=(5, 5))
    A_sym = symmetrize(A)

    assert A_sym is A
    assert_array_equal(A_sym, A_sym.T)


def test_symmetrize_batched() -> None:
    from donk.utils import symmetrize

    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, 5, 5))
    A_sym = symmetrize(A)

    assert A_sym is A
    assert A_sym.shape == (3, 5, 5)
    assert_array_equal(A_sym[0], A_sym[0].T)
    assert_array_equal(A_sym[1], A_sym[1].T)
    assert_array_equal(A_sym[2], A_sym[2].T)


def test_regularize() -> None:
    from donk.utils import regularize

    rng = np.random.default_rng(0)
    A = rng.normal(size=(5, 5))
    A_copy = A.copy()
    A_reg = regularize(A, 1e-6)

    assert A_reg is A
    assert_array_equal(A_reg, A_copy + 1e-6 * np.eye(5))


def test_regularize_batched() -> None:
    from donk.utils import regularize

    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, 5, 5))
    A_copy = A.copy()
    A_reg = regularize(A, 1)

    assert A_reg is A
    assert_array_equal(A_reg, A_copy + 1 * np.eye(5))
    assert_array_equal(A, A_copy + 1 * np.eye(5))


def test_regularize_slice_1() -> None:
    from donk.utils import regularize

    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, 5, 5))
    A_copy = A.copy()
    regularize(A[1], 1e-2)

    assert_array_equal(A[0], A_copy[0])
    assert_array_equal(A[1], A_copy[1] + 1e-2 * np.eye(5))
    assert_array_equal(A[2], A_copy[2])


def test_regularize_slice_2() -> None:
    from donk.utils import regularize

    rng = np.random.default_rng(0)
    A = rng.normal(size=(5, 5))
    A_copy = A.copy()
    A_reg = regularize(A[:3, :3], 1e-2)

    assert A_reg.base is A
    assert_array_equal(A_reg, A_copy[:3, :3] + 1e-2 * np.eye(3))
    assert_array_equal(A[:3, :3], A_copy[:3, :3] + 1e-2 * np.eye(3))
    assert_array_equal(A[3:, :3], A_copy[3:, :3])
    assert_array_equal(A[:3, 3:], A_copy[:3, 3:])
    assert_array_equal(A[3:, 3:], A_copy[3:, 3:])


def test_trace_of_prod() -> None:
    from donk.utils import trace_of_product

    rng = np.random.default_rng(0)
    A = rng.normal(size=(5, 6))
    B = rng.normal(size=(6, 5))
    assert_allclose(trace_of_product(A, B), np.trace(A @ B))


def test_trace_of_prod_batched() -> None:
    from donk.utils import trace_of_product

    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, 5, 6))
    B = rng.normal(size=(3, 6, 5))

    for i in range(3):
        assert_allclose(trace_of_product(A, B)[i], np.trace(A[i] @ B[i]))

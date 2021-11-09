from __future__ import annotations

import numpy as np

from donk.utils import trace_of_product


class QuadraticCosts():
    """Quadratic cost function.

    cost(x, u) = 1/2 [x u]^T*C*[x u] + [x u]^T*c + cc
    """

    def __init__(self, C: np.ndarray, c: np.ndarray, cc: np.ndarray):
        """Initialize this LinearDynamics object.

        Args:
            C: (..., dX+dU, dX+dU), Quadratic term
            c: (..., dX+dU), Linear term
            cc: (...,), Constant term

        """
        T = c.shape[:-1]
        dXU = c.shape[-1]

        # Check shapes
        assert C.shape == (*T, dXU, dXU), f"{C.shape} != {(*T, dXU, dXU)}"
        assert c.shape == (*T, dXU), f"{c.shape} != {(*T, dXU)}"
        assert cc.shape == (*T, ), f"{cc.shape} != {(*T, )}"

        self.C = C
        self.c = c
        self.cc = cc

    def __add__(self, other) -> QuadraticCosts:
        """Sum two cost functions."""
        if isinstance(other, QuadraticCosts):
            return QuadraticCosts(self.C + other.C, self.c + other.c, self.cc + other.cc)
        return NotImplemented

    def __mul__(self, other) -> QuadraticCosts:
        """Scale cost function with  vonstant scalar."""
        if np.isscalar(other):
            return QuadraticCosts(self.C * other, self.c * other, self.cc * other)
        return NotImplemented

    def __rmul__(self, other) -> QuadraticCosts:
        """Scale cost function with  vonstant scalar."""
        return self.__mul__(other)

    def compute_costs(self, traj: np.ndarray) -> np.ndarray:
        """Evaluate costs for trajectories.

        Args:
            traj: (..., dX+dU), State-action tuples

        Returns:
            costs: (...,), Costs at each time step
        """
        costs = (
            # Quadratic costs
            # 1/2 * mu^t C mu
            np.einsum("...i,...ij,...j->...", traj, self.C, traj) / 2 +
            # Linear costs
            # mu^T c
            np.sum(traj * self.c, axis=-1) +
            # Constant costs
            # cc
            self.cc
        )
        return costs

    def expected_costs(self, traj_mean: np.ndarray, traj_covar: np.ndarray) -> np.ndarray:
        """Compute estimated costs for trajectory distribution under quadratic cost approximation.

        1/2 * (Tr(C Sigma) + mu^t C mu) + mu^T c + cc

        Args:
            traj_mean: (..., dXU), Mean of trajectory distribution
            traj_covar: (..., dXU), Covariances of trajectory distribution

        Returns:
            expectation: (..., ) Expected costs at each time step
        """
        expectation = self.compute_costs(traj_mean) + trace_of_product(self.C, traj_covar) / 2
        return expectation

    @staticmethod
    def quadratic_cost_approximation_l2(t: np.ndarray, w: np.ndarray) -> QuadraticCosts:
        """Constructs a quadratic cost function for a non-equilibrium L2 loss.

        L(xu) = sum_i w_i*(t_i-xu_i)**2

        Actually, this is not an approximation. Rather it's exact.

        Args:
            t: (..., dXU), Target for each state/action at each timestep.
            w: (..., dXU), Weight for each state/action at each timestep.
        """
        dX = t.shape[-1]

        C = np.einsum("...i,ij->...ij", w, np.eye(dX))
        c = -np.einsum("...i,...i->...i", w, t)
        cc = np.einsum("...i,...i,...i->...", t, w, t) / 2

        return QuadraticCosts(C, c, cc)

    @staticmethod
    def quadratic_cost_approximation_l1(xu: np.ndarray, t: np.ndarray, w: np.ndarray, alpha: float) -> QuadraticCosts:
        """Constructs a quadratic cost function approximation for a non-equilibrium L1 loss.

        L(xu) = sum_i w_i*sqrt((t_i-xu_i)**2 + alpha)

        For sake of differentiability, a small constant alpha is added.

        Args:
            xu: (..., dX+dU), Current trajectory
            t: (..., dX+dU), Target for each state/action at each timestep.
            w: (..., dX+dU), Weight for each state/action at each timestep.
            alpha: Small constant to create smooth differentials
        """
        dXU = xu.shape[-1]

        abs_squared = (t - xu)**2 + alpha
        abs_d = np.sqrt(abs_squared)
        abs_d_cubed = abs_squared * abs_d

        # C = diag(alpha * w / )
        C = np.einsum("...i,ij->...ij", alpha * w / abs_d_cubed, np.eye(dXU))
        c = w * ((xu - t) / abs_d - alpha * xu / abs_d_cubed)
        cc = np.sum(w * (abs_d + xu * (t - xu) / abs_d + alpha * xu**2 / abs_d_cubed / 2), axis=-1)

        return QuadraticCosts(C, c, cc)

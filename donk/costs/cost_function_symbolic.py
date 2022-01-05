from __future__ import annotations

from collections.abc import Callable

import numpy as np

from donk.costs.cost_function import CostFunction
from donk.costs.quadratic_costs import QuadraticCosts


class SymbolicCostFunction(CostFunction):
    """CostFunction using symbolic differentiation using SymPy to approximate arbitrary functions."""

    def __init__(self, cost_fun: Callable[[int, np.ndarray, np.ndarray | None], float], T: int, dX: int, dU: int) -> None:
        """Initialize this SymbolicCostFunction.

        Args:
            cost_fun: The primitive cost function. A callable which evaluates the costs for x and u at a given time step t.
            T: Time horizon
            dX: Dimension of state space
            dU: Dimension of action space
        """
        from sympy import Matrix, diff, symbols, lambdify

        self.T, self.dX, self.dU = T, dX, dU

        a = np.array(symbols(f"a:{dX+dU}"))

        self.C = []
        self.c = []
        self.cc = []
        for t in range(T + 1):
            loss = cost_fun(t, a[:dX], a[dX:] if t < T else None)
            loss_d = np.array([diff(loss, a[i]) for i in range(dX + dU)])
            loss_dd = np.array([[diff(loss_d[j], a[i]) for j in range(dX + dU)] for i in range(dX + dU)])

            # Lambdify sympy expressions for performance
            self.C.append(lambdify([a if t < self.T else a[:dX]], Matrix(loss_dd)))
            self.c.append(lambdify([a if t < self.T else a[:dX]], Matrix(loss_d - a @ loss_dd)))
            self.cc.append(lambdify([a if t < self.T else a[:dX]], loss - a @ loss_d + a.T @ loss_dd @ a / 2))

        self.a = a

    def quadratic_approximation(self, X: np.ndarray, U: np.ndarray) -> QuadraticCosts:
        """Compute a quadratic approximation (2nd order Taylor) at the given trajectory.

        Args:
            X: (T+1, dX), states
            U: (T, dX), actions
        """
        # Check shapes
        assert X.shape == (self.T + 1, self.dX), f"{X.shape} != {(self.T + 1, self.dX)}"
        assert U.shape == (self.T, self.dU), f"{U.shape} != {(self.T , self.dU)}"

        C = []
        c = []
        cc = []

        # Numerical evaluations for each timestep
        for t in range(self.T + 1):
            XU = np.concatenate([X[t], U[t]]) if t < self.T else X[t]
            C.append(self.C[t](XU))
            c.append(self.c[t](XU))
            cc.append(self.cc[t](XU))

        return QuadraticCosts(
            C=np.array(C).astype(np.float64),
            c=np.array(c).reshape((self.T + 1, self.dX + self.dU)).astype(np.float64),
            cc=np.array(cc).astype(np.float64),
        )

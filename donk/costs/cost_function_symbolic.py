from __future__ import annotations

from collections.abc import Callable

import numpy as np

from donk.costs.cost_function import CostFunction
from donk.costs.quadratic_costs import QuadraticCosts


class SymbolicCostFunction(CostFunction):
    """CostFunction using symbolic differentiation using SymPy to approximate arbitrary functions."""

    def __init__(self, cost_fun: Callable[[np.ndarray, np.ndarray], np.ndarray], T: int, dX: int, dU: int) -> None:
        """Initialize this SymbolicCostFunction.

        Args:
            cost_fun: The primitive cost function. A callable which evaluates the costs for a trajectory at each timestep.
                      Returns an ndarray with shape (T, )
            T: Time horizon
            dX: Dimension of state space
            dU: Dimension of action space
        """
        from sympy import Matrix, diff, symbols, lambdify

        self.T, self.dX, self.dU = T, dX, dU

        X_sym = np.array(symbols(f"x:{(T+1)*dX}")).reshape(T + 1, dX)
        U_sym = np.array(symbols(f"u:{T*dU}")).reshape(T, dU)
        costs = cost_fun(X_sym, U_sym)

        self.C = []
        self.c = []
        self.cc = []
        for t in range(T + 1):
            XU = np.concatenate([X_sym[t], U_sym[t]]) if t < T else X_sym[t]

            loss = costs[t]
            loss_d = np.array([diff(loss, xu) for xu in XU])
            loss_dd = np.array([[diff(loss_d[j], xu) for j in range(len(XU))] for xu in XU])

            # Lambdify sympy expressions for performance
            self.C.append(lambdify([X_sym, U_sym], Matrix(loss_dd)))
            self.c.append(lambdify([X_sym, U_sym], Matrix(loss_d - XU @ loss_dd)))
            self.cc.append(lambdify([X_sym, U_sym], loss - XU @ loss_d + XU.T @ loss_dd @ XU / 2))

    def quadratic_approximation(self, X: np.ndarray, U: np.ndarray) -> QuadraticCosts:
        """Compute a quadratic approximation (2nd order Taylor) at the given trajectory.

        Args:
            X: (T+1, dX), states
            U: (T, dX), actions
        """
        T, dX, dU = self.T, self.dX, self.dU

        # Check shapes
        assert X.shape == (T + 1, dX), f"{X.shape} != {(T + 1, dX)}"
        assert U.shape == (T, dU), f"{U.shape} != {(T , dU)}"

        # Allocate space
        C = np.empty((T + 1, dX + dU, dX + dU))
        c = np.empty((T + 1, dX + dU))
        cc = np.empty((T + 1, ))

        # Numerical evaluations for each timestep
        for t in range(T):
            C[t] = self.C[t](X, U)
            c[t] = self.c[t](X, U).flatten()
            cc[t] = self.cc[t](X, U)
        # Final state
        C[T, :dX, :dX] = self.C[T](X, U)
        c[T, :dX] = self.c[T](X, U).flatten()
        cc[T] = self.cc[T](X, U)

        return QuadraticCosts(C, c, cc)

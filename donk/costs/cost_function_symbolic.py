from __future__ import annotations

from collections.abc import Callable
from typing import Dict, List

import numpy as np

from donk.costs.cost_function import CostFunction
from donk.costs.quadratic_costs import QuadraticCosts


def _vectorize_cost_function(fun):
    return np.vectorize(lambda X, U: np.array(fun(X, U)), signature="(v,x),(t,u)->(v)")


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
        from sympy import Matrix, diff, lambdify, symbols

        self.T, self.dX, self.dU = T, dX, dU

        # Create symbols
        X_sym = np.array(symbols(f"x:{(T+1)*dX}")).reshape(T + 1, dX)
        U_sym = np.array(symbols(f"u:{T*dU}")).reshape(T, dU)

        # Eval costs
        costs = cost_fun(X_sym, U_sym)

        # Store base cost function as vectorized to allow broadcasting
        self.cost_fun = _vectorize_cost_function(lambdify([X_sym, U_sym], list(costs)))

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
        cc = np.empty((T + 1,))

        # Numerical evaluations for each timestep
        for t in range(T):
            C[t] = self.C[t](X, U)
            c[t] = self.c[t](X, U).flatten()
            cc[t] = self.cc[t](X, U)
        # Final state
        C[T] = 0  # Zero action potion
        C[T, :dX, :dX] = self.C[T](X, U)
        c[T, dX:] = 0  # Zero action potion
        c[T, :dX] = self.c[T](X, U).flatten()
        cc[T] = self.cc[T](X, U)

        return QuadraticCosts(C, c, cc)

    def compute_costs(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Evaluate costs for trajectories.

        Args:
            X: (..., T+1, dX), states
            U: (..., T, dU), actions

        Returns:
            costs: (..., T+1), Costs at each time step
        """
        return self.cost_fun(X, U)


class MultipartSymbolicCostFunction(SymbolicCostFunction):
    """SymbolicCostFunction composed of named partial cost functions."""

    def __init__(
        self, cost_funs: List[Callable[[np.ndarray, np.ndarray], np.ndarray]], cost_function_names: List[str], T: int, dX: int, dU: int
    ) -> None:
        from sympy import lambdify, symbols

        def cost_fun(X, U):
            """Sum up individual parts."""
            cost = cost_funs[0](X, U)
            for fn in cost_funs[1:]:
                cost += fn(X, U)
            return cost

        super().__init__(cost_fun, T, dX, dU)

        # Create symbols
        X_sym = np.array(symbols(f"x:{(T+1)*dX}")).reshape(T + 1, dX)
        U_sym = np.array(symbols(f"u:{T*dU}")).reshape(T, dU)

        # Lambdify and vectorize cost functions
        self.cost_funs = [_vectorize_cost_function(lambdify([X_sym, U_sym], list(cf(X_sym, U_sym)))) for cf in cost_funs]

        self.cost_function_names = cost_function_names

    def compute_costs_individual(self, X: np.ndarray, U: np.ndarray) -> Dict[str, np.ndarray]:
        """Evaluate costs for trajectories.

        Args:
            X: (..., T+1, dX), states
            U: (..., T, dU), actions

        Returns:
            costs: {name: (..., T+1)}, Costs at each time step, of each sub-cost function
        """
        return {name: cf(X, U) for cf, name in zip(self.cost_funs, self.cost_function_names)}

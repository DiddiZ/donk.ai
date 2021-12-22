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
        from sympy import Matrix, diff, symbols

        self.T, self.dX, self.dU = T, dX, dU

        a = np.array(symbols(f"a:{dX+dU}"))

        self.C = []
        self.c = []
        self.cc = []
        for t in range(T + 1):
            loss = cost_fun(t, a[:dX], a[dX:] if t < T else None)
            loss_d = np.array([diff(loss, a[i]) for i in range(dX + dU)])
            loss_dd = np.array([[diff(loss_d[j], a[i]) for j in range(dX + dU)] for i in range(dX + dU)])

            self.C.append(Matrix(loss_dd))
            self.c.append(Matrix(loss_d - a @ loss_dd))
            self.cc.append(loss - a @ loss_d + a.T @ loss_dd @ a / 2)

        self.a = a

    def quadratic_approximation(self, X: np.ndarray, U: np.ndarray) -> QuadraticCosts:
        """Compute a quadratic approximation (2nd order Taylor) at the given trajectory.

        Args:
            X: (T+1, dX), states
            U: (T, dX), actions
        """
        # Check shapes
        assert X.shape == (self.T + 1, self.dX), f"{X.shape} != {(self.T + 1, self.dX)}"
        assert U.shape == (self.T, self.dX), f"{U.shape} != {(self.T , self.dX)}"

        C = []
        c = []
        cc = []

        for t in range(self.T + 1):
            substitutions = {a: x for a, x in zip(self.a[:self.dX], X[t])}
            if t < self.T:
                substitutions.update({a: u for a, u in zip(self.a[self.dX:], U[t])})

            C.append(self.C[t].subs(substitutions).evalf())
            c.append(self.c[t].subs(substitutions).evalf())
            cc.append(self.cc[t].subs(substitutions).evalf())

        return QuadraticCosts(
            C=np.array(C).astype(np.float64),
            c=np.array(c).reshape((self.T + 1, self.dX + self.dU)).astype(np.float64),
            cc=np.array(cc).astype(np.float64),
        )

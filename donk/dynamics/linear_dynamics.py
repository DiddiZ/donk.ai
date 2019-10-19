class LinearDynamics:
    """Stochastic lienar dynamics model."""

    def __init__(self, Fm, fv, dyn_covar):
        """Initialize this LinearDynamics object.

        Args:
            Fm: (T, dX, dX + dU), linear term
            fv: (T, dX), constant term
            dyn_covar: (T, dX, dX), covariances

        """
        self.T, self.dX = fv.shape
        self.dU = Fm.shape[2] - self.dX

        self.Fm = Fm
        self.fv = fv
        self.dyn_covar = dyn_covar

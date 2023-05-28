import numpy as np

from donk.utils.batched import batched_cholesky, batched_inv_spd


class TimeVaryingLinearGaussian:
    """Time-varying linear model with Gaussian noise.

    `p(y|t,x) ~ N(coeffcients_t * x + intercept_t, covar_t)`
    """

    def __init__(
        self, coefficients: np.ndarray, intercept: np.ndarray, covar: np.ndarray = None, inv_covar: np.ndarray = None
    ) -> None:
        """Initialize this TimeVaryingLinearGaussian object.

        Must provide either covariance or precision, or both.

        Args:
            coefficients: (T, dY, dX), Linear term
            intercept: (T, dY), Constant term
            covar: (T, dY, dY), Covariances
            inv_covar: (T, dY, dY), Inverse covariances, precision.
        """
        if covar is None and inv_covar is None:
            raise ValueError("Must provide covar or inv_covar.")

        self.T, dY, dX = coefficients.shape

        # Check shapes
        assert coefficients.shape == (self.T, dY, dX), f"{coefficients.shape} != {(self.T, dY, dX )}"
        assert intercept.shape == (self.T, dY), f"{intercept.shape} != {(self.T, dY)}"
        if covar is not None:
            assert covar.shape == (self.T, dY, dY), f"{covar.shape} != {(self.T, dY, dY)}"
        if inv_covar is not None:
            assert inv_covar.shape == (self.T, dY, dY), f"{inv_covar.shape} != {(self.T,dY, dY)}"

        self._coefficients = coefficients
        self._intercept = intercept
        self._covar = covar
        self._chol_covar = None
        self._inv_covar = inv_covar

    @property
    def coefficients(self) -> np.ndarray:
        """Linear coefficients of this model.

        Returns:
            coefficients: (T, dY, dX)
        """
        return self._coefficients

    @property
    def intercept(self) -> np.ndarray:
        """Constant intercepts of this model.

        Returns:
            intercept: (T, dY)
        """
        return self._intercept

    @property
    def covar(self) -> np.ndarray:
        """Covariances of this model.

        Computed lazily, if required.

        Returns:
            covar: (T, dY, dY)
        """
        if self._covar is None:
            self._covar = batched_inv_spd(batched_cholesky(self._inv_covar))
        return self._covar

    @property
    def chol_covar(self) -> np.ndarray:
        """Cholesky decoposition of the covariances of this model.

        Computed lazily, if required.

        Returns:
            chol_covar: (T, dY, dY)
        """
        if self._chol_covar is None:
            self._chol_covar = batched_cholesky(self.covar)
        return self._chol_covar

    @property
    def inv_covar(self) -> np.ndarray:
        """Inverse covariances/precisions of this model.

        Computed lazily, if required.

        Returns:
            inv_covar: (T, dY, dY)
        """
        if self._inv_covar is None:
            self._inv_covar = batched_inv_spd(self.chol_covar)
        return self._inv_covar

    def predict(self, X: np.ndarray, t: int = None, noise: np.ndarray = None) -> np.ndarray:
        """Sample preditions from this model.

        Samples the Gaussian distributing given by `p(y|t,x) ~ N(coeffcients_t * x + intercept_t, covar_t)`.

        Can either sample a specific timestep, or from all.

        Noise can be omitted to compute the output expectations.

        Args:
            X: (..., dX) or (..., T, dX) Inputs
            t: Supply to sample from a specific timestept, optional
            noise: (..., dY,) Output noise, optional

        Returns:
            Y: (..., dY) or  (..., T, dY) Outputs
        """
        if t is not None:
            Y = np.einsum("ux,...x->...u", self.coefficients[t], X) + self.intercept[t]
            if noise is not None:
                Y += np.einsum("ij,...j->...i", self.chol_covar[t], noise)
        else:
            dX = self.coefficients.shape[-1]
            if X.shape[-2:] != (self.T, dX):
                raise ValueError(f"Input must have shape (..., {self.T}, {dX}), not {X.shape}")
            Y = np.einsum("tyx,...tx->...ty", self.coefficients, X) + self.intercept
            if noise is not None:
                Y += np.einsum("tij,...tj->...ti", self.chol_covar, noise)
        return Y

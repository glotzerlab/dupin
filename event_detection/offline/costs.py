"""Implements cost functions for use in event detection."""
import numpy as np
import ruptures as rpt
import scipy as sp


class CostLinearFit(rpt.base.BaseCost):
    r"""Compute the L1 cumulative error of piecewise linear fits in time.

    Used to compute the relative cumulative L1 deviation from a linear fit
    piecewise fit of a signal (potentially multidimensional).

    .. math::

        C(s, e) = \\min\\limits_{m, b} \sum_i |y_i - (m x_i + b)|

    :math:`m` and :math:`b` can be vectors in the case of a multidimensional
    signal (the summation also goes across dimensions.

    Note:
        For use in ``ruptures`` search algorithms. To use properly `fit` must be
        called first with the signal.
    """

    model = "linear_regression"
    min_size = 2

    def __init__(self):
        """Create a CostLinearFit object."""
        pass

    def fit(self, signal: np.ndarray):
        """Store signal and compute base errors for later cost checking."""
        if len(signal.shape) == 1:
            self._signal = np.ascontiguousarray(signal.reshape((1, -1)))
        else:
            self._signal = np.ascontiguousarray(signal.T)

        if len(signal.shape) == 2:
            self._base_errors = np.ones(signal.shape[1], dtype=float)
        else:
            self._base_errors = np.ones(1, dtype=float)
        self._base_errors = self._individual_errors(0, len(signal))
        self._x = np.linspace(0, 1, len(self._signal), dtype=float)

    def error(self, start: int, end: int):
        """Return the cost for signal[start:end]."""
        return sum(self._individual_errors(start, end))

    def _individual_errors(self, start: int, end: int):
        errors = []
        x = self._x[start:end]
        for base_error, signal in zip(self._base_errors, self._signal):
            prefactor = (end - start) / base_error
            y = signal[start:end]
            linear_regression = sp.stats.linregress(x, y)
            errors.append(prefactor * self._l1(linear_regression, x, y))
        return errors

    @staticmethod
    def _l1(linear_regression, x: np.ndarray, y: np.ndarray):
        predicted_y = linear_regression.slope * x + linear_regression.intercept
        return np.sum(np.abs(predicted_y - y))

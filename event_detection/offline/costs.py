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

    def fit(self, signal):
        """Store signal and compute base errors for later cost checking."""
        self._signal = signal

        if len(signal.shape) == 2:
            self._base_errors = [1.0 for _ in range(signal.shape[1])]
        else:
            self._base_errors = [1.0]
        self._base_errors = self._individual_errors(0, len(signal))

    def error(self, start, end):
        """Return the cost for signal[start:end]."""
        return sum(self._individual_errors(start, end))

    def _individual_errors(self, start, end):
        errors = []
        x = np.linspace(0, 1, end - start)
        for ind_base_error, signal in zip(self._base_errors, self._signal.T):
            linear_regression = sp.stats.linregress(x, signal[start:end])
            prefactor = (end - start) / ind_base_error
            errors.append(
                prefactor * self._l1(linear_regression, x, signal[start:end])
            )
        return errors

    @staticmethod
    def _l1(linear_regression, x, y):
        predicted_y = linear_regression.slope * x + linear_regression.intercept
        return np.sum(np.abs(predicted_y - y))

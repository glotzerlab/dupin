"""Implements cost functions for use in event detection."""

import numpy as np
import ruptures as rpt
import scipy as sp
from sklearn import preprocessing


class CostLinearFit(rpt.base.BaseCost):
    r"""Compute the L1 cumulative error of piecewise linear fits in time.

    Used to compute the relative cumulative L1 deviation from a linear piecewise
    fit of a signal.

    .. math::

        C(s, e) = \\min\\limits_{m, b} \sum_i |y_i - (m x_i + b)|

    :math:`m` and :math:`b` can be vectors in the case of a multidimensional
    signal (the summation also goes across dimensions.

    Parameters
    ----------
    metric : str, optional
        What metric to use in computing the error. Defaults to `"l1"`. Options
        are `"l1"` and `"l2"`.

    Note:
        For use in ``ruptures`` search algorithms. To use properly `fit` must be
        called first with the signal.
    """

    model = "linear_regression"
    min_size = 3
    _metrics = {"l1", "l2"}

    def __init__(self, metric="l1"):
        """Create a CostLinearFit object."""
        if metric not in self._metrics:
            raise ValueError(f"Available metrics are {self._metrics}.")
        self._metric = "_" + metric

    def fit(self, signal: np.ndarray):
        """Store signal and compute base errors for later cost checking."""
        if len(signal.shape) == 1:
            signal = signal.reshape((-1, 1))
        signal = preprocessing.MinMaxScaler().fit_transform(signal)
        self._signal = np.ascontiguousarray(signal.T)
        self._x = np.linspace(0, 1, self._signal.shape[1], dtype=float)

    def error(self, start: int, end: int):
        """Return the cost for signal[start:end]."""
        return sum(self._individual_errors(start, end))

    def _individual_errors(self, start: int, end: int):
        errors = []
        x = self._x[start:end]
        for y in self._signal[:, start:end]:
            regression = sp.stats.linregress(x, y)
            predicted_y = regression.slope * x + regression.intercept
            errors.append(getattr(self, self._metric)(y, predicted_y))
        return errors

    @staticmethod
    def _l1(y: np.ndarray, predicted_y: np.ndarray):
        return np.sum(np.abs(predicted_y - y))

    @staticmethod
    def _l2(y: np.ndarray, predicted_y: np.ndarray):
        return np.sqrt(np.sum(np.sq(predicted_y - y)))

    @property
    def signal(self) -> np.ndarray:
        """numpy.ndarray: Required by Ruptures to exist in \
                (N_samples, N_dimensions)."""
        return self._signal.T

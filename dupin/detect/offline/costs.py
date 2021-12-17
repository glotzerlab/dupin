"""Implements cost functions for use in event detection."""

import numpy as np
import ruptures as rpt
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
        self._metric = getattr(self, "_" + metric)

    def _compute_cumsum(self, arr: np.ndarray):
        """Compute a cumulative sum for use in computing partial sums.

        Only works for 1 or 2D arrays.
        """
        if arr.ndim > 1:
            shape = arr.shape[:-1] + (arr.shape[-1] + 1,)
        else:
            shape = len(arr) + 1
        out = np.empty(shape)
        out[..., 0] = 0
        np.cumsum(arr, out=out[..., 1:], axis=arr.ndim - 1)
        return out

    def fit(self, signal: np.ndarray):
        """Store signal and compute base errors for later cost checking."""
        if len(signal.shape) == 1:
            signal = signal.reshape((-1, 1))
        y = preprocessing.MinMaxScaler().fit_transform(signal).T
        self._y = y
        self._x = np.linspace(0, 1, y.shape[1], dtype=float)
        self._x_cumsum = self._compute_cumsum(self._x)
        self._x_sq_cumsum = self._compute_cumsum(self._x ** 2)
        self._xy_cumsum = self._compute_cumsum(self._x[None, :] * y)
        self._y_cumsum = self._compute_cumsum(y)

    def error(self, start: int, end: int):
        """Return the cost for signal[start:end]."""
        m, b = self._get_regression(start, end)
        predicted_y = m[:, None] * self._x[None, start:end] + b[:, None]
        return self._metric(self._y[:, start:end], predicted_y)

    def _get_regression(self, start: int, end: int):
        """Compute a least squared regression on each dimension.

        Though never explicitly done the computations follow an X martix with a
        1 vector prepended to allow for a non-zero intercept.
        """
        # Given their off by one nature no offset for computing the partial sum
        # in needed.
        sum_x = self._x_cumsum[end] - self._x_cumsum[start]
        sum_x_sq = self._x_sq_cumsum[end] - self._x_sq_cumsum[start]
        sum_xy = self._xy_cumsum[:, end] - self._xy_cumsum[:, start]
        sum_y = self._y_cumsum[:, end] - self._y_cumsum[:, start]
        N = end - start
        m = (N * sum_xy - (sum_x * sum_y)) / (N * sum_x_sq - (sum_x * sum_x))
        b = (sum_y - (m * sum_x)) / N
        return m, b

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
        return self._y.T


class CostLinearBiasedFit(CostLinearFit):
    """Compute a start to end linear fit and pentalize error and bias."""

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

    @property
    def signal(self) -> np.ndarray:
        """numpy.ndarray: Required by Ruptures to exist in \
                (N_samples, N_dimensions)."""
        return self._signal.T

    def _individual_errors(self, start: int, end: int):
        errors = []
        x = self._x[start:end]
        for y in self._signal[:, start:end]:
            slope = self._get_slope(x, y)
            intercept = self._get_intercept(x[0], y[0], slope)
            predicted_y = slope * x + intercept
            errors.append(getattr(self, self._metric)(y, predicted_y))
        return errors

    @staticmethod
    def _get_slope(x: np.ndarray, y: np.ndarray) -> float:
        return (y[-1] - y[0]) / (x[-1] - x[0])

    @staticmethod
    def _get_intercept(x: float, y: float, slope: float) -> float:
        return y - slope * (x)

    @staticmethod
    def _l1(y: np.ndarray, predicted_y: np.ndarray):
        diff = predicted_y - y
        base_error = np.sum(np.abs(diff))
        return (1 + np.sum(diff) / base_error) * base_error

    @staticmethod
    def _l2(y: np.ndarray, predicted_y: np.ndarray):
        diff = predicted_y - y
        base_error = np.sqrt(np.sum(np.sq(diff)))
        return (1 + np.sum(diff) / np.sum(np.abs(diff))) * base_error

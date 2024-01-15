"""Implements cost functions for use in event detection."""

from abc import ABC, abstractmethod

import numpy as np
import ruptures as rpt
from sklearn import preprocessing


class BaseLinearCost(rpt.base.BaseCost, ABC):
    """Base class for costs using linear fits of features across signal."""

    _metrics = frozenset(("l1", "l2"))
    min_size = 3

    def __init__(self, metric="l1"):
        """Create a CostLinearFit object."""
        if metric not in self._metrics:
            msg = f"Available metrics are {self._metrics}."
            raise ValueError(msg)
        self._metric = getattr(self, "_" + metric)

    def fit(self, signal: np.ndarray):
        """Store signal and compute base errors for later cost checking."""
        if len(signal.shape) == 1:
            signal = signal.reshape((-1, 1))
        self._y = preprocessing.MinMaxScaler().fit_transform(signal).T
        self._x = np.linspace(0, 1, self._y.shape[1], dtype=float)

    def error(self, start: int, end: int):
        """Return the cost for signal[start:end]."""
        m, b = self._get_regression(start, end)
        predicted_y = self._get_predicted(m, b, start, end)
        return self._metric(self._y[:, start:end], predicted_y)

    def _get_predicted(
        self, m: np.ndarray, b: np.ndarray, start: int, end: int
    ):
        return m[:, None] * self._x[None, start:end] + b[:, None]

    @abstractmethod
    def _get_regression(self, start: int, end: int):
        pass

    @staticmethod
    def _l1(y: np.ndarray, predicted_y: np.ndarray):
        return np.sum(np.abs(predicted_y - y))

    @staticmethod
    def _l2(y: np.ndarray, predicted_y: np.ndarray):
        return np.sqrt(np.sum(np.square(predicted_y - y)))

    @property
    def signal(self) -> np.ndarray:
        """:math:`(N_{samples}, N_{dim})` numpy.ndarray of float: signal \
           fitted on.
        """  # noqa: D205
        return self._y.T


class CostLinearFit(BaseLinearCost):
    r"""Compute the L1 cumulative error of piecewise linear fits in time.

    Works with `ruptures`_. Used to compute the relative cumulative L1 deviation
    from a linear piecewise fit of a signal.

    .. math::

        C(s, e) = \min\limits_{m, b} \sum_i |y_i - (m x_i + b)|

    :math:`m` and :math:`b` can be vectors in the case of a multidimensional
    signal (the summation also goes across dimensions.

    Parameters
    ----------
    metric : `str`, optional
        What metric to use in computing the error. Defaults to ``"l1"``. Options
        are ``"l1"`` and ``"l2"``.

    Note:
        For use in `ruptures`_ search algorithms. To use properly `fit` must be
        called first with the signal.
    """

    model = "linear_regression"

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
        super().fit(signal)
        self._x_cumsum = self._compute_cumsum(self._x)
        self._x_sq_cumsum = self._compute_cumsum(self._x**2)
        self._xy_cumsum = self._compute_cumsum(self._x[None, :] * self._y)
        self._y_cumsum = self._compute_cumsum(self._y)

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


# TODO: Fill out documentation.
class CostLinearBiasedFit(CostLinearFit):
    """Compute a start to end linear fit and pentalize error and bias.

    Works with `ruptures`_.
    """

    model = "biased_linear_regression"

    def _get_regression(
        self, start: int, end: int
    ) -> tuple[np.ndarray, np.ndarray]:
        m = (self._y[:, end - 1] - self._y[:, start]) / (
            self._x[None, end - 1] - self._x[None, start]
        )
        b = self._y[:, start] - (m * self._x[None, start])
        return m, b

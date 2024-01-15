"""Implements offline methods for detecting events in molecular simulations."""

import logging
import warnings
from typing import Callable, Optional, Union

import kneed as kd
import numpy as np
import pandas as pd
import ruptures as rpt

_logger = logging.getLogger(__name__)

ElbowDetector = Callable[[list[float]], int]


class SweepDetector:
    """Detects the optimal number of change points in a time series.

    By using a composed change point detection algorithm that detects a fixed
    number of change points and an elbow detection algorithm, this class detects
    the optimal change points as defined by the change points detected at the
    elbow of the cost versus number of change points plot.

    Parameters
    ----------
    detector: Union[``ruptures.base.BaseEstimator``, \
                    ``callable`` [[`numpy.ndarray`, `int`], \
                            `tuple` [`list` [`int` ], `float` ]]:
        The detector to use for each round of change point detection. Can be any
        callable which takes in a NumPy array signal of shape
        :math:`(N_{frames}, N_{features})` and the number of change points and
        returns a tuple containing the list of change points and the total cost
        for the change points. The argument can also be any of `ruptures`_
        estimators.
    max_change_points: int
        The maximum number of change points to attempt to detect.
    elbow_detector: ``callable`` [[ `list` [ `float` ]], `int`], optional
        A callable that takes in a list of costs and outputs the elbow of the
        data. The callable should return ``None`` if no elbow can be detected.
        Defaults to the KNEEDLE algorithm provided by the kneedle package (see
        `kneedle_elbow_detection` for dupin defaults).
    tolerance: `float`, optional
        The percentile change in cost below which to stop detecting higher
        numbers of change points. Since detecting :math:`n+1` change points is
        by definition going to decrease the cost less than the last iteration,
        this is a reliable way to prevent wasted computation. For instance, a
        value of 0.01 means that if adding a change point decreases the cost by
        less than one percent of the previous value the detector stops
        immediately regardless of ``max_change_points``.
    """

    def __init__(
        self,
        detector: Union[
            rpt.base.BaseEstimator,
            Callable[[np.ndarray, int], tuple[list[int], float]],
        ],
        max_change_points: int,
        elbow_detector: Optional[ElbowDetector] = None,
        tolerance: float = 1e-3,
    ) -> None:
        if isinstance(detector, rpt.base.BaseEstimator):
            self._detector = _RupturesWrapper(detector)
        else:
            self._detector = detector
        self.max_change_points = max_change_points
        if elbow_detector is None:
            self._elbow_detector = kneedle_elbow_detection
        else:
            self._elbow_detector = elbow_detector
        self.tolerance = tolerance

    def fit(self, data: np.ndarray) -> list[int]:
        """Fit and return change points for given data.

        Compute the change points for ``[0, self.max_change_points]``, and
        detect the elbow of the associated costs if any.

        Parameters
        ----------
        data: numpy.ndarray
            The data to detect change points for.

        Returns
        -------
        list[int]
            The change points if any. An empty list means no change points were
            detected.
        """
        if isinstance(data, pd.DataFrame):
            return self.fit(data.to_numpy())
        change_points, penalties = self._get_change_points(data)
        self.costs_ = penalties
        self.change_points_ = change_points

        if len(self.costs_) > 1:
            elbow_index = self._elbow_detector(penalties)
        else:
            elbow_index = None
        if elbow_index is None:
            self.opt_n_change_points_ = 0
            self.opt_change_points_ = []
            return self.opt_change_points_

        self.opt_n_change_points_ = elbow_index
        self.opt_change_points_ = change_points[elbow_index]
        return self.opt_change_points_

    def _get_change_points(
        self, data: np.ndarray
    ) -> tuple[list[int], list[float]]:
        penalties = []
        change_points = []
        # Get the base level pentalty of the entire sequence
        points, cost = self._detector(data, 0)
        change_points.append(points)
        penalties.append(cost)
        for num_change_points in range(1, self.max_change_points + 1):
            points, cost = self._detector(data, num_change_points)
            # None indicates that the detection failed, and we should return
            # with the currenlty detected change points.
            if points is None:
                break
            penalties.append(cost)
            change_points.append(points)
            # Check if cost changed enough to justify finding an additional
            # change point.
            if abs(cost - penalties[-2]) / penalties[-2] <= self.tolerance:
                break
        return change_points, penalties


def kneedle_elbow_detection(
    costs: list[float],
    S: int = 1,
    interp_method: str = "interp1d",
    curve: str = "convex",
    direction: str = "decreasing",
    **kwargs,
):
    r"""Run the KNEEDLE algorithm for elbow detection from the kneed package.

    Note:
        See the `kneed`_ documentation for more information on parameter
        selection in the KNEEDLE algorithm.

    Parameters
    ----------
    costs : `list` [`float` ]
        The list/array of costs along some implicit x.
    S : `int`, optional
        A sensitivity parameter. Higher values require more obvious
        elbows/knees, while the lowest value, 1, will detect elbows soonest.
        Defaults to 1.
    interp_method: `str`, optional
        The method of interpolation for the discrete points. Options are
        "interp1d" and "polynomial". "interp1d" uses
        `scipy.interpolate.interp1d`, and "polynomial" uses `numpy.polyfit`.
        Defualts to "interp1d".
    curve: `str`, optional
        Will detect knees if "concave" and elbows if "convex". Defaults to
        "convex".
    direction: `str`, optional
        Either "increasing" or "decreasing". Whether the trend from left to
        right is increasing or decreasing. Defaults to "decreasing".
    \*\*kwargs : dict
        Other keyword arguments to pass to ``kneed.KneeLocator``.

    Returns
    -------
    int
        The predicted index for the elbow.
    """
    with warnings.catch_warnings(record=True) as record:
        warnings.filterwarnings("ignore", module=".*kneed.*")
        detector = kd.KneeLocator(
            x=range(0, len(costs)),
            y=costs,
            S=S,
            interp_method=interp_method,
            curve=curve,
            direction=direction,
            **kwargs,
        )
        if len(record) > 0:
            _logger.info("No knee/elbow found.")
    return detector.elbow


def two_pass_elbow_detection(
    threshold: int, detector: Optional[ElbowDetector] = None
) -> ElbowDetector:
    """Return a two pass function of another elbow detection algorithm.

    The detector runs a first pass of the elbow detector ``detector`` and
    determines if the elbow is far enough along the cost curve (determined by
    ``threshold``). If it is not, the detector runs a second pass with only the
    points at or beyond the first pass's estimated elbow. This is designed to
    help with the case where the first elbow detected is expected to be from
    such a prodigious decrease in the cost function that the proper number of
    events would not be detected such as smaller events within a phase
    transition.

    Note:
        If the second pass returns ``None`` the first pass elbow will still be
        used.

    Parameters
    ----------
    threshold : int
        If the first pass of the elbow detector computes an elbow less than
        threshold, run a second pass. Otherwise, the detector just returns the
        first pass.
    detector : ``callable`` [[`list` [`float` ]], `int`], optional
        The callable to use for both sweeps of elbow detection. Defaults to
        `kneedle_elbow_detection`.

    Returns
    -------
    ``callable`` [[`list` [`float`], `list` [`float` ]], `int`]
        Returns a new elbow detector that uses the two steps scheme shown above.
    """
    if detector is None:
        detector = kneedle_elbow_detection

    def find_elbow(costs: list[float]) -> int:
        first_pass = detector(costs)
        if first_pass is None:
            _logger.debug(
                "No elbow detected in first pass of two_pass_elbow_detection."
            )
            return None
        if first_pass < threshold:
            second_pass = detector(costs[first_pass:])
            if second_pass is None:
                _logger.debug(
                    "No second elbow detected in second pass of "
                    "two_pass_elbow_detection."
                )
                return first_pass
            return second_pass + first_pass
        return first_pass

    return find_elbow


class _RupturesWrapper:
    def __init__(self, detector: rpt.base.BaseEstimator):
        self.detector = detector
        self.data = None
        self._previous_detections = {}

    def __call__(self, data: np.ndarray, n_change_points: int):
        if self.data is None or self.data is not data:
            self.fit(data)
        return self.memonized_detect(n_change_points)

    def fit(self, data: np.ndarray):
        self.data = data
        self.detector.fit(data)
        self._previous_detections = {}

    def detect(self, n_change_points: int) -> tuple[list[int], float]:
        if n_change_points == 0:
            return [], self.detector.cost.error(0, len(self.data))
        # An AssertionError is raised if no suitable change point can be found
        # by a decector. We return (None, 0) to indicate the failure.
        try:
            change_points = self.detector.predict(n_change_points)
        except rpt.exceptions.BadSegmentationParameters as err:
            _logger.info(
                f"Error detecting {n_change_points} change points. "
                f"Original error: {type(err).__name__}({err!s})."
            )
            return (None, 0)
        # Return None if the correct number of change points were not detected
        # which can happen when no change point addition reduces cost. This
        # should not happen on a well defined cost function but numerical errors
        # happen.
        if len(change_points) != n_change_points + 1:
            return (None, 0)
        # The selection of change points was successful, compute costs and
        # return
        cost = self.detector.cost.sum_of_costs(change_points)
        # Remove the last index of the signal from the sequence of change points
        change_points.pop()
        return (change_points, cost)

    def memonized_detect(self, n_change_points: int) -> tuple[list[int], float]:
        if n_change_points in self._previous_detections:
            return self._previous_detections[n_change_points]
        self._previous_detections[n_change_points] = self.detect(
            n_change_points
        )
        return self._previous_detections[n_change_points]

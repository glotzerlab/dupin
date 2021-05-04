"""Implements offline methods for detecting events in molecular simulations."""
from typing import Callable, List, Optional, Tuple, Union

import kneed as kd
import numpy as np
import ruptures as rpt

ElbowDetector = Callable[[List[float]], int]


class SweepDetector:
    """Detects the optimal number of change points in a time series.

    By using a composed change point detection algorithm that detects a fixed
    number of change points and an elbow detection algorithm, this class detects
    the optimal change points as defined by the change points detected at the
    elbow of the cost versus number of change points plot.
    """

    def __init__(
        self,
        detector: Union[
            rpt.base.BaseEstimator,
            Callable[[np.ndarray, int], Tuple[List[int], float]],
        ],
        max_change_points: int,
        elbow_detector: Optional[ElbowDetector] = None,
    ) -> None:
        """Create a SweepDetector object."""
        self._detector = detector
        self.max_change_points = max_change_points
        if elbow_detector is None:
            self._elbow_detector = kneedle_elbow_detection
        else:
            self._elbow_detector = elbow_detector

    def fit(self, data: np.ndarray) -> List[int]:
        """Fit and return change points for given data."""
        change_points, penalties = self._get_change_points(data)
        self.costs_ = penalties
        self.change_points_ = change_points

        opt_n_change_points = self._get_elbow(penalties)
        if opt_n_change_points is None:
            self.opt_n_change_points_ = 0
            self.opt_n_change_points_ = []
            return self.opt_change_points_

        self.opt_n_change_points_ = opt_n_change_points
        self.opt_change_points_ = change_points[opt_n_change_points - 1]
        return self.opt_change_points_

    def _get_change_points(
        self, data: np.ndarray
    ) -> Tuple[List[int], List[float]]:
        penalties = []
        change_points = [[]]
        # Get the base level pentalty of the entire sequence
        if isinstance(self._detector, rpt.base.BaseEstimator):
            self._detector.cost.fit(data)
            penalties.append(self._detector.cost.error(0, len(data)))
        else:
            penalties.append(self._detector(data, 0))
        for num_change_points in range(1, self.max_change_points + 1):
            if isinstance(self._detector, rpt.base.BaseEstimator):
                points = self._detector.fit_predict(
                    data, n_bkps=num_change_points
                )
                cost = self._detector.cost.sum_of_costs(points)
            else:
                points, cost = self._detector(data, num_change_points)
            penalties.append(cost)
            change_points.append(points)
        return change_points, penalties

    def _get_elbow(self, costs: List[float]) -> int:
        elbow = self._elbow_detector(costs)
        if elbow is None:
            return len(costs) + 1
        return elbow


def kneedle_elbow_detection(costs: List[float], **kwargs):
    r"""Run the KNEEDLE algorithm for elbow detection.

    Parameters
    ----------
    costs : list[float]
        The list/array of costs along some implicit x.
    \*\*kwargs : dict
        keyword arguments to pass to ``kneed.KneeLocator``.

    Returns
    -------
    int
        The predicted index for the elbow.
    """
    kwargs.setdefault("S", 1.0)
    kwargs.setdefault("interp_method", "interp1d")
    x = range(0, len(costs))
    detector = kd.KneeLocator(
        x=x, y=costs, curve="convex", direction="decreasing", **kwargs
    )
    return detector.elbow

"""Classes for transforming array quantities into scalar features."""

from abc import abstractmethod
from collections.abc import Callable
from typing import Dict, Optional, Tuple

import numpy as np


class ArrayReducer(Callable):
    """Base class for implementing schemes to reduce distribution into features.

    Requires the implemnation of `__call__`.
    """

    @abstractmethod
    def __call__(self, distribution: np.ndarray) -> Dict[str, float]:
        """Turn a distribution into scalar features.

        Parameters
        ----------
        distribution : np.ndarray
            The array representing a distribution to reduce.

        Return
        -------
        signals : dict[str, float]
            Returns a dictionary with string keys representing the type of
            reduction for its associated value. For instance if the value is the
            max of the distribution, a logical key value would be ``'max'``. The
            key only needs to represent the reduction, the original distribution
            name will be dealt with by a generator.
        """
        pass

    def update(self, state):
        """Perform necessary updates to the internals with new system state.

        This is primarly for use when the filter needs system specific
        information to do its reduction.
        """
        pass


class Percentile(ArrayReducer):
    """Reduce a distribution into percentile values."""

    def __init__(self, percentiles: Optional[Tuple[int]] = None) -> None:
        """Create a `Percentile` object.

        Parameters
        ----------
        percentiles : tuple[int], optional
            The percentiles in integer form (i.e. 100% equals 100). By defualt,
            every 10% increment from 0% to 100% (inclusive) is taken.
        """
        if percentiles is None:
            self._percentiles = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)

    def __call__(self, distribution: np.ndarray) -> Dict[str, float]:
        """Return the signals with dd% keys."""
        N = len(distribution) - 1
        sorted_distribution = np.sort(distribution)
        indices = np.rint(np.multiply(self._percentiles, N / 100)).astype(int)
        return {
            f"{percent}%": sorted_distribution[i]
            for i, percent in zip(indices, self._percentiles)
        }


class SpatialAveraging(ArrayReducer):
    """Composite Reducer that first spatially averages the given distribution.

    To do this, the update method is used to update neighbors.
    """

    def __init__(self, final_filters: Tuple[ArrayReducer]):
        """Create a `SpatialAveraging` object.

        Parameters
        ----------
        final_filters : tuple[ArrayReducer]
            The reducers to use after spatial averaging. These are necessary
            since in general spatially averaging will still produce a
            distribution (just coarse grained).
        """
        self._filters = final_filters

    def __call__(self, distribution: np.ndarray) -> Dict[str, float]:
        """Return the signals from the final filters.

        Prepends "spatial_averaged" to the signal `dict` keys.
        """
        spatial_average = self._spatially_average(distribution)
        return {
            "-".join(("spatial_averaged", key)): signal
            for filter_ in self._filters
            for key, signal in filter_(spatial_average)
        }

    def update(self, state):
        """Update system neighbors."""
        # TODO
        pass

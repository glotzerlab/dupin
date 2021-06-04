"""Classes for transforming array quantities into scalar features."""

from abc import abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, Dict, Optional, Tuple, Union

import freud
import numba
import numpy as np

from .util import _state_to_freud_system, _state_to_id


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
            percentiles = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
        self._percentiles = percentiles

    def __call__(self, distribution: np.ndarray) -> Dict[str, float]:
        """Return the signals with dd% keys."""
        return {
            f"{percent}%": value
            for percent, value in zip(
                self._percentiles,
                np.percentile(distribution, self._percentiles),
            )
        }


class NthGreatest(ArrayReducer):
    """Reduce a distribution to the Nth greatest values."""

    def __init__(self, indices: Tuple[int]) -> None:
        """Create a `NthGreatest` object.

        Parameters
        ----------
        indices : tuple[int], optional
            The indices to query. Negative indices are the Nth smallest values.
            Zero is not smallest value in the array.
        """
        self._indices = indices

    def __call__(self, distribution: np.ndarray) -> Dict[str, float]:
        """Return the signals with modified keys."""
        sorted_array = np.sort(distribution)
        return {
            f"{index}-th-{'greatest' if index > 0 else 'least'}": sorted_array[
                -index
            ]
            for index in self._indices
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
            for key, signal in filter_(spatial_average).items()
        }

    @abstractmethod
    def update(self, state):
        """Update system neighbors."""
        pass

    @abstractmethod
    def _spatially_average(self, distribution: np.ndarray) -> np.ndarray:
        """Return the spatially averaged distribution.

        Parameters
        ----------
        distribution : np.ndarray
            The array representing a distribution to reduce.

        Return
        ------
        np.ndarray
            The spatially averaged distribution. Usually this should be an array
            smaller or the same size as the original array. Currently no
            locality preserving method of spatial averaging is available (i.e.
            mantaining information about where in the simulation each point in
            the averaged distribution comes from is impossible).
        """
        pass


@numba.njit
def _freud_neighbor_summing(
    arr: np.ndarray, particle_index: np.ndarray, neighbor_index: np.ndarray
) -> np.ndarray:
    summed_array = np.zeros(arr.shape)
    for i, j in zip(particle_index, neighbor_index):
        summed_array[i] += arr[j]
    return summed_array


class NeighborAveraging(SpatialAveraging):
    r"""Average distribution about neighboring data points.

    Uses freud's neighbor finding to allow for generic spatial averaging across
    a system respecting periodic boundary conditions. Just like freud's neighbor
    finding three modes are available: an :math:`\epsilon`-ball approach which
    finds all neighbors within a set radius, a set number of nearest neighbors,
    and using the neighbors of a Voronoi tesselation.

    In general, you should not exclude self-neighbors as this will make the
    averaging not include the central particle in the averaging.
    """

    def __init__(
        self,
        final_filters: Tuple[ArrayReducer],
        neighbor_kwargs: Union[Dict[str, Any], str],
    ):
        """Create a `SpatialAveraging` object.

        Parameters
        ----------
        final_filters : tuple[ArrayReducer]
            The reducers to use after spatial averaging. These are necessary
            since in general spatially averaging will still produce a
            distribution (just coarse grained).
        neighbor_kwargs : Union[dict[str, Any], str]
            The arguments to pass to the `freud.locality.AABBQuery`. The
            exception is when `neighbor_kwargs` is the string ``'voronio'`` in
            which case the Voronio tesselation is used for neighbor finding.
        """
        self._filters = final_filters
        if isinstance(neighbor_kwargs, Mapping):
            self._neighbor_kwargs = neighbor_kwargs
            self._use_voronoi = False
        else:
            self._neighbor_kwargs = None
            self._use_voronoi = True
        self._state_id = -1

    def update(self, state):
        """Update system neighbors."""
        new_id = _state_to_id(state)
        if new_id == self._state_id:
            return
        self._state_id = new_id
        system = _state_to_freud_system(state)
        if self._use_voronoi:
            voronio = freud.locality.Voronoi()
            voronio.compute(system)
            nlist = voronio.nlist
        else:
            query = freud.locality.AABBQuery(*system)
            nlist = query.query(
                system[1], self._neighbor_kwargs
            ).toNeighborList()
        # Must store arrays explicitly or some garbage collection/memory error
        # occurs leaving the array size and data corrupted and crashes the
        # program.
        self._nlist_array = nlist[:]
        self._neighbor_counts = nlist.neighbor_counts

    def _spatially_average(self, distribution: np.ndarray) -> np.ndarray:
        return _freud_neighbor_summing(
            distribution, self._nlist_array[:, 0], self._nlist_array[:, 1]
        ) / self._neighbor_counts.astype(float)

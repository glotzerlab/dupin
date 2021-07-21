"""Spatial averaging DataMap classes."""

import numba
import numpy as np
import numpy.typing

from . import base


@numba.njit
def _freud_neighbor_summing(
    arr: np.ndarray, particle_index: np.ndarray, neighbor_index: np.ndarray
) -> np.ndarray:
    summed_array = np.zeros(arr.shape)
    for i, j in zip(particle_index, neighbor_index):
        summed_array[i] += arr[j]
    return summed_array


class NeighborAveraging(base.DataMap):
    r"""Average distribution about neighboring data points.

    Uses neighbors for spatial averaging across a system. Neighbors can be
    passed in manually in :math:`(i, j)` pairs through a tuple of arrays or
    through a freud neighbor list.

    In general, you should not exclude self-neighbors as this will make the
    averaging not include the central particle in the averaging.

    Note:
        This class does not support wrapping the call signature of the composed
        generator like objet, since it modifies the call signature.
    """

    def __init__(
        self,
        generator: base.GeneratorLike,
        expected_kwarg: str = "spatial_neighbors",
        remove_kwarg: bool = True,
    ):
        """Create a `SpatialAveraging` object.

        Parameters
        ----------
        generator: GeneratorLike
            A generator like object to reduce.
        expected_kwarg: str, optional
            The expected key word argument passed to `__call__` to use as
            neighbors. Defaults to "spatial_neighbors".
        remove_kwargs: bool, optional
            Whether the specified ``expected_kwarg`` should be removed before
            passing through to the composed generators.
        """
        self._expected_kwarg = expected_kwarg
        self._remove_kwarg = remove_kwarg
        super().__init__(generator)

    def __call__(self, *args, **kwargs):
        """Call the underlying generator performing the spatial averaging.

        The call signature is the same as the composed generator with the
        addition of the ``expected_kwarg``.
        """
        self._current_neighbors = kwargs[self._expected_kwarg]
        if self._remove_kwarg:
            del kwargs[self._expected_kwarg]
        return super().__call__(*args, **kwargs)

    def compute(self, data: np.typing.ArrayLike) -> np.typing.ArrayLike:
        """Perform spatial averaging using provided neighbors.

        Parameters
        ----------
        distribution : :math:`(N,)` np.ndarray of float
            The array representing a distribution to spatially average.

        Return
        -------
        signals : dict[str, np.ndarray]
            A dictionary with the key "spatially_averaged" and spatially
            averaged distribution as a value.
        """
        if isinstance(self._current_neighbors, tuple):
            i_index, j_index = self._current_neighbors
        # Assume freud nlist
        else:
            i_index = self._current_neighbors.point_indices
            j_index = self._current_neighbors.query_point_indices
        return {
            "spatially_averaged": _freud_neighbor_summing(
                data, i_index, j_index
            )
        }

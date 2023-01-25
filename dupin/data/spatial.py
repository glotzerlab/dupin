"""Spatial averaging DataMap classes."""

import numpy as np
import numpy.typing as npt

from . import base


def _njit(*args, **kwargs):
    """Allow for JIT when numba is found."""
    try:
        import numba
    except ImportError:
        return lambda x: x
    return numba.njit(*args, **kwargs)


@_njit
def _freud_neighbor_summing(
    arr: np.ndarray,
    particle_index: np.ndarray,
    neighbor_index: np.ndarray,
    base: np.ndarray,
) -> np.ndarray:
    for i, j in zip(particle_index, neighbor_index):
        base[i] += arr[j]


class NeighborAveraging(base.DataMap):
    r"""Average distribution about neighboring data points.

    Uses neighbors for spatial averaging across a system. Neighbors can be
    passed in manually in :math:`(i, j)` pairs through a tuple of arrays or
    through a freud neighbor list.

    Warning:
        The correct setting of ``exclude_ii`` is important for correct results.
        The original particle's data should be included once in the averaging.
        Incorrect setting can lead to the original data not being included at
        all or twice.

    Note:
        This class does not support wrapping the call signature of the composed
        generator like objet, since it modifies the call signature.
    """

    def __init__(
        self,
        expected_kwarg: str = "spatial_neighbors",
        remove_kwarg: bool = True,
        exclude_ii: bool = True,
    ):
        """Create a `NeighborAveraging` object.

        Parameters
        ----------
        expected_kwarg: `str`, optional
            The expected key word argument passed to
            `dupin.data.base.DataModifier.__call__` to use as neighbors.
            Defaults to "spatial_neighbors".
        remove_kwargs: `bool`, optional
            Whether the specified ``expected_kwarg`` should be removed before
            passing through to the composed generators.
        exclude_ii: `bool`, optional
            Whether the passed neighbor list will excludes ``ii`` interactions.
            Defaults to ``True``. If set incorrectly this will cause erroneous
            results.
        """
        self._expected_kwarg = expected_kwarg
        self._remove_kwarg = remove_kwarg
        self._exclude_ii = exclude_ii
        super().__init__()

    def update(self, args, kwargs):
        """Call the underlying generator performing the spatial averaging.

        The call signature is the same as the composed generator with the
        addition of the ``expected_kwarg``.
        """
        nlist = kwargs[self._expected_kwarg]
        if isinstance(nlist, tuple):
            i_index, j_index = self._current_neighbors
            counts = np.unique(i_index, return_counts=True)[1]
        # Assume freud nlist
        else:
            i_index = nlist.point_indices
            j_index = nlist.query_point_indices
            counts = np.copy(nlist.neighbor_counts)

        if self._exclude_ii:
            counts += 1
        self._i_index = i_index
        self._j_index = j_index
        self._counts = counts
        if self._remove_kwarg:
            del kwargs[self._expected_kwarg]
        return args, kwargs

    def compute(self, data: npt.ArrayLike) -> npt.ArrayLike:
        """Perform spatial averaging using provided neighbors.

        Parameters
        ----------
        distribution : :math:`(N,)` numpy.ndarray of float
            The array representing a distribution to spatially average.

        Return
        -------
        signals : dict[str, numpy.ndarray]
            A dictionary with the key "spatially_averaged" and spatially
            averaged distribution as a value.
        """
        if self._exclude_ii:
            averaged_data = np.copy(data)
        else:
            averaged_data = np.zeros(data.shape)
        _freud_neighbor_summing(
            data, self._i_index, self._j_index, averaged_data
        )
        return {"spatially_averaged": averaged_data / self._counts}

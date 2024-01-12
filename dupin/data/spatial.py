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


@_njit()
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
    passed in manually in an array of :math:`(i, j)` pairs, through a tuple of
    arrays, or through a `freud` neighbor list.

    Parameters
    ----------
    expected_kwarg: `str`, optional
        The expected key word argument passed to
        `dupin.data.base.DataModifier.__call__` to use as neighbors.
        Defaults to "spatial_neighbors".
    remove_kwargs: `bool`, optional
        Whether the specified ``expected_kwarg`` should be removed before
        passing through to the composed generators. Defaults to ``True``.
    excluded_self: `bool`, optional
        Whether the passed neighbor lists will exclude self neighbors.
        ``True`` means self neighbors will be added by the instance, and
        ``False`` means the neighbor list provides self neighbors. Defaults
        to ``True``. If set incorrectly this will cause erroneous
        results (double or no counting).

    Warning:
        A particle should be listed as its own neighbor for purposes of the
        averaging. So if the passed neighbor list does not include self
        neighbors ``excluded_self`` should be true.

    Note
    ----
        This class can remove neighbors from the call signature for generators
        or maps that don't require it through the ``remove_kwarg`` argument.

    Note
    ----
        The neighbors must be passed as a keyword argument for
        `NeighborAveraging` to recognize it.
    """

    def __init__(
        self,
        expected_kwarg: str = "spatial_neighbors",
        remove_kwarg: bool = True,
        excluded_self: bool = True,
    ):
        self._expected_kwarg = expected_kwarg
        self._remove_kwarg = remove_kwarg
        self._excluded_self = excluded_self
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

        if self._excluded_self:
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
        if self._excluded_self:
            averaged_data = np.copy(data)
        else:
            averaged_data = np.zeros(data.shape)
        _freud_neighbor_summing(
            data, self._i_index, self._j_index, averaged_data
        )
        return {"spatially_averaged": averaged_data / self._counts}

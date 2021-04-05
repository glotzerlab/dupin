"""Implements classes and methods to compute steinhardt/minkowski OP signals."""
from typing import Any, Dict, List, Union

import freud
import numba
import numpy as np

from .generate import Generator
from .reduce import ArrayReducer
from .util import _state_to_freud_system


@numba.vectorize(
    [numba.float64(numba.complex128), numba.float32(numba.complex64)]
)
def _conj_sq(x):
    return (x.real * x.real) + (x.imag * x.imag)


@numba.njit
def _to_steinhardt(js, harmonics, l, Np, weights):  # noqa: E741
    avg_qlm = np.zeros(harmonics.shape[1], dtype=np.complex64)
    steinhardt = np.zeros((Np, l + 1), dtype=np.float32)
    cum_weight = 0.0
    cur_particle = 0
    for i, (j, weight) in enumerate(zip(js, weights)):
        if cur_particle != j:
            avg_qlm /= cum_weight
            _reduce_harmonics(avg_qlm, l, steinhardt[cur_particle])
            cum_weight = 0.0
            cur_particle += 1
            avg_qlm[:] = 0

        avg_qlm += weight * harmonics[i, :]
        cum_weight += 1.0

    avg_qlm /= cum_weight
    _reduce_harmonics(avg_qlm, l, steinhardt[cur_particle])
    return steinhardt


@numba.njit(inline="always")
def _reduce_harmonics(avg_qlm, l, out):  # noqa: E741
    start = 0
    end = 0
    for cur_l in range(l + 1):
        end += cur_l * 2 + 1
        prefactor = 4 * np.pi / (2 * cur_l + 1)
        out[cur_l] = np.sqrt(prefactor * np.sum(_conj_sq(avg_qlm[start:end])))
        start = end


class SteinhardtGenerator(Generator):
    """Generate steinhardt order parameters for signals."""

    def __init__(
        self,
        max_l: int,
        reducers: List[ArrayReducer],
        neighbors: Union[Dict[str, Any], str, freud.locality.NeighborList],
    ):
        """Create a SteinhardtGenerator object."""
        self.max_l = max_l
        self.reducers = reducers
        self.neighbors = neighbors
        self.compute = freud.environment.LocalDescriptors(max_l)

    def generate(self, state) -> Dict[str, float]:
        """Return the output signal for a given state.

        Parameters
        ----------
        state: state-like object
            An object with a `hoomd.Snapshot` like API. Examples include
            `gsd.hoomd.Frame` and `hoomd.Snapshot`. This is used to pass to
            generator to return the corresponding signals.

        Returns
        -------
        signals:
            Returns a mapping of signal names to floating point values.
        """
        for reducer in self.reducers:
            reducer.update(state)
        system = _state_to_freud_system(state)
        if isinstance(self.neighbors, str) and self.neighbors == "voronoi":
            voronoi = freud.locality.Voronoi()
            voronoi.compute(system)
            nlist = voronoi.nlist
        else:
            query = freud.locality.AABBQuery(*system)
            nlist = query.query(system[1], self.neighbors).toNeighborList()
        self.compute.compute(system, neighbors=nlist)
        steinhardt = _to_steinhardt(
            nlist[:, 0],
            self.compute.sph,
            self.max_l,
            nlist.num_points,
            nlist.weights,
        )
        signals = {}
        for i, q in enumerate(steinhardt.T):
            signals.update(
                {
                    "-".join((key, f"$Q_{{{i}}}$")): value
                    for reducer in self.reducers
                    for key, value in reducer(q).items()
                }
            )
        return signals

"""Interface from freud to event_detection."""

import copy
from collections.abc import Sequence
from typing import Dict, List, Optional, Tuple, Union

try:
    import freud
except ImportError:

    class _ModuleError:
        def __init__(self, module):
            self.module = module

        def __getattribute__(self, attr):
            raise RuntimeError(
                f"This feature is not available as the module "
                f"{object.__getattribute__(self, 'module')} is not available."
            )

    freud = _ModuleError("freud")

import numpy as np

from .generate import Generator
from .reduce import ArrayReducer
from .util import _state_to_freud_system


class FreudDescriptorDefinition:
    """Defines the interface between freud and event_detection."""

    def __init__(
        self,
        compute,
        attrs: Union[str, Tuple[str, ...]],
        reducers: Optional[Tuple[ArrayReducer]] = None,
        drop_kwargs: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        r"""Create a `FreudDescriptorDefinition` object.

        Parameters
        ----------
        compute: freud.util._Compute
            freud objects with a ``compute`` method.
        attrs: str or Sequence[str] or dict[str, str]
            A mapping of attribute names to desired signal names. If the value
            in a entry is ``None`` the key value is used. A single string or
            sequence of strings can be passed and will be converted to the
            appropriate dict instance.
        reducers: Sequence[signal.ArrayReducer], optional
            A sequence of `signal.ArrayReducer` objects to use for creating
            features from distributions or array quantities. All reducers are
            applied to all ``attrs``. If any attributes specified are arrays,
            and this list is empty then an error is produced.
        drop_kwargs: Sequence[str], optional
            List of keyword arguments to ignore when passed from a
            `FreudDescriptors` object.
        \*\*kwargs:
            mapping of keyword arguments to pass to ``compute.compute``.

        Note:
            If ``'neighbors'`` is used as a key-word argument, the string
            ``'voronoi'`` can be passed as its value to indicate that the
            Voronoi tesselation should be used for determining neighbors.
        """
        self.compute = compute
        if isinstance(attrs, str):
            self.attrs = {attrs: attrs}
        elif isinstance(attrs, Sequence):
            self.attrs = {attr: None for attr in attrs}
        else:
            self.attrs = attrs
        self.reducers = reducers
        self.drop_kwargs = drop_kwargs if drop_kwargs is not None else {}
        self.kwargs = kwargs

    def __call__(self, system, **kwargs):
        """Return signals generated from this descriptor."""
        for reducer in self.reducers:
            reducer.update(system)
        for key in self.drop_kwargs:
            kwargs.pop(key, None)
        kwargs.update(self.kwargs)
        if kwargs["neighbors"] == "voronoi":
            voronoi = freud.locality.Voronoi()
            voronoi.compute(system)
            kwargs["neighbors"] = voronoi.nlist
        self.compute.compute(system, **kwargs)
        signals = {}
        for attr in self.attrs:
            signals.update(self._process_attr(attr))
        return signals

    def _process_attr(self, attr):
        if (name := self.attrs[attr]) is None:
            name = attr
        data = getattr(self.compute, attr)
        if isinstance(data, np.ndarray):
            if self.reducers is None:
                raise RuntimeError(
                    f"Cannot process array quantity "
                    f"{attr} without a filter."
                )
            return {
                "-".join((key, name)): value
                for reducer in self.reducers
                for key, value in reducer(data).items()
            }
        else:
            return {name: data}


class FreudDescriptors(Generator):
    """Wraps `freud` compute objects for use in generating signals.

    When a returned quantity returned by an instances internal compute is an
    array this is assumed to be of a consistent size. That is if it returns an
    array of size 100, then it will always be of size 100.
    """

    def __init__(
        self,
        computes: List[FreudDescriptorDefinition],
        reducers: Optional[Tuple[ArrayReducer]] = None,
        **kwargs,
    ):
        r"""Create a `FreudDescriptors` object.

        Parameters
        ----------
        compute: freud.util._Compute
            The list of `FreudDescriptorDefinition` objects to generate the
            signals from. Plain freud compute objects can also be passed and
            they will be wrapped around a `FreudDescriptorDefinition`. This
            assumes the desired attribute is ``particle_order``.
        reducers: Sequence[signal.ArrayReducer]
            A sequence of `signal.ArrayReducer` objects to use for creating
            features from distributions or array quantities. All reducers are
            applied to all ``attrs``. If any attributes specified are arrays,
            and this list is empty then an error is produced.
        \*\*kwargs:
            mapping of keyword arguments to pass to the individual
            `FreudDescriptorDefinition` objects' call method.

        Note:
            If ``'neighbors'`` is used as a key-word argument, the neighbor
            query specified will be precomputed for use in all the descriptors.
            This is desirable if all/most of the computes have the same neighbor
            arguments, and undesirable for heterogenious neighbor querying.
            Also, if the string ``'voronoi'`` can be passed as its value to
            indicate that the Voronoi tesselation should be used for determining
            neighbors.
        """
        self._computes = []
        for compute in computes:
            if isinstance(compute, FreudDescriptorDefinition):
                if compute.reducers is None:
                    compute.reducers = reducers
                self._computes.append(compute)
            else:
                self._computes.append(
                    FreudDescriptorDefinition(
                        compute, "particle_order", reducers
                    )
                )
        self._kwargs = kwargs

    def generate(self, state) -> Dict[str, float]:
        """Generate the specified signals from the internal freud compute.

        state: state-like object
            An object with a `hoomd.Snapshot` like API. Examples include
            `gsd.hoomd.Frame` and `hoomd.Snapshot`. This is used to pass to
            generator to return the corresponding signals.
        """
        system = _state_to_freud_system(state)
        kwargs = copy.copy(self._kwargs)
        if "neighbors" in kwargs:
            if (
                isinstance(kwargs["neighbors"], str)
                and kwargs["neighbors"] == "voronoi"
            ):
                voronoi = freud.locality.Voronoi()
                voronoi.compute(system)
                kwargs["neighbors"] = voronoi.nlist
            else:
                query = freud.locality.AABBQuery(*system)
                kwargs["neighbors"] = query.query(
                    system[1], kwargs["neighbors"]
                ).toNeighborList()

        signals = {}
        for compute in self._computes:
            signals.update(compute(system, **kwargs))
        return signals

"""`data.base.DataMap` subclasses to transform distributional data."""

from typing import Callable, Dict, List, Union

import numpy as np

from . import base


class Identity(base.DataMap):
    """A identity mapping for use in a data pipeline."""

    def __init__(self, generator: base.GeneratorLike):
        """Create a Identity object.

        This DataMap does nothing to the data passed.

        Parameters
        ----------
        generator: GeneratorLike
            A generator like object to reduce.
        """
        self._generator = generator

    def compute(self, data: np.typing.ArrayLike) -> np.typing.ArrayLike:
        """Do nothing to the base distribution.

        Parameters
        ----------
        distribution : :math:`(N,)` np.ndarray of float
            The array representing a distribution to map.

        Return
        -------
        signals : dict[str, float]
            Returns a dictionary a ``None`` key and the passed in data as a
            value.
        """
        return {None: data}


class Tee(base.DataMap):
    """Enable mutliple maps to act on the same generator like object."""

    def __init__(
        self,
        generator: base.GeneratorLike,
        maps: List[Callable[[base.GeneratorLike], base.DataMap]],
    ):
        """Create a data.reduce.Tee object.

        Parameters
        ----------
        generator: GeneratorLike
            A generator like object to map to another distribution.
        reducers: list[callable[base.GeneratorLike, base.DataReducer]]
            A sequence of callables that take a generator like object and
            returns a data map. Using the ``wraps`` class method with a
            `DataMap` subclass is a useful combination.
        """
        self._maps = [map_(generator) for map_ in maps]
        super().__init__(generator)

    def compute(
        self, distribution: np.typing.ArrayLike
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Run all composed map computes."""
        processed_data = {}
        for map_ in self._maps:
            processed_data.update(map_.compute(distribution))
        return processed_data

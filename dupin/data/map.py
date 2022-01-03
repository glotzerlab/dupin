"""`data.base.DataMap` subclasses to transform distributional data."""

from typing import Callable, Dict, List, Union

import numpy as np
import numpy.typing as npt

from . import base


class Identity(base.DataMap):
    """A identity mapping for use in a data pipeline."""

    def __init__(self, generator: base.GeneratorLike):
        """Create a Identity object.

        This DataMap does nothing to the data passed.

        Parameters
        ----------
        generator: generator_like
            A generator like object to reduce.
        """
        super().__init__(generator)

    def compute(self, data: npt.ArrayLike) -> npt.ArrayLike:
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
        generator: generator_like
            A generator like object to map to another distribution.
        reducers: list[callable[generator_like, base.DataReducer]]
            A sequence of callables that take a generator like object and
            returns a data map. Using the ``wraps`` class method with a
            `DataMap` subclass is a useful combination.
        logger: dupin.data.logging.Logger
            A logger object to store data from the data pipeline for individual
            elements of the composed maps.
        """
        self._maps = [map_(generator) for map_ in maps]
        super().__init__(generator)

    def compute(
        self, distribution: npt.ArrayLike
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Run all composed map computes."""
        processed_data = {}
        for map_ in self._maps:
            processed_data.update(map_.compute(distribution))
        return processed_data

    def update(self, args, kwargs):
        """Iterate over composed mappings and call update."""
        for map_ in self._maps:
            args, kwargs = map_.update(args, kwargs)
        return args, kwargs

    def attach_logger(self, logger):
        """Add a logger to this step in the data pipeline.

        Parameters
        ----------
        logger: dupin.data.logging.Logger
            A logger object to store data from the data pipeline for individual
            elements of the composed maps.
        """
        self._logger = logger
        for map_ in self._maps:
            try:
                map_.attach_logger(logger)
            # Do nothing if generator does not have attach_logger logger
            # function (e.g. custom map function).
            except AttributeError:
                pass

    def remove_logger(self):
        """Remove a logger from this step in the pipeline if it exists."""
        self._logger = None
        for map_ in self._maps:
            try:
                map_.remove_logger()
            # Do nothing if generator does not have remove_logger logger
            # function (e.g. custom map function).
            except AttributeError:
                pass


CustomMap = base.CustomMap

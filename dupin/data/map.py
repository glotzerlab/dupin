"""`dupin.data.base.DataMap` subclasses to transform distributional data.

Mapping in dupin is the idea of taking one distribution and transforming it
into another. This is distinct from the mathematical view of functions as maps
and is more general than Python's map builtin. A distribution/array can be
mapped to a new array of any size. A common example in molecular simulations is
spatial averaging which reduces local fluctuations of features. This particular
map can be found in `dupin.data.spatial.NeighborAveraging`.
"""

from typing import Union

import numpy as np
import numpy.typing as npt

from . import base


class Identity(base.DataMap):
    """A identity mapping for use in a data pipeline.

    This class maps a distribution to itself. This is useful when using with
    `Tee`.

    Example::

        generator.pipe(
            du.data.map.Tee([
                du.data.map.Identity()
                du.data.map.CustomMap(lambda x: {"new_dist": x + 2})
            ])
        )
    """

    def __init__(self):
        super().__init__()

    def compute(self, data: npt.ArrayLike) -> npt.ArrayLike:
        """Return the same distribution.

        Parameters
        ----------
        distribution : :math:`(N,)` numpy.ndarray of float
            The array representing a distribution to map.

        Return
        -------
        signals : dict[str, float]
            Returns a dictionary with the key ``None`` and the passed in data as
            a value.
        """
        return {None: data}


class Tee(base.DataMap):
    """Combine mutliple maps into one acting on the same generator object.

    Example::

        generator.pipe(
            du.data.map.Tee([
                du.data.map.Identity(),
                du.data.map.CustomMap(lambda x: {"new_dist": x + 2})
            ])
        )

    Parameters
    ----------
    reducers: `list` [`dupin.data.base.DataReducer`]
        A sequence of data modifiers.
    """

    def __init__(
        self,
        maps: list[base.DataMap],
    ):
        if len(maps) == 0:
            msg = "Cannot have empty maps sequence."
            raise ValueError(msg)
        self._maps = maps
        super().__init__()

    def compute(
        self, distribution: npt.ArrayLike
    ) -> dict[str, Union[float, np.ndarray]]:
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

    def _decorate(self, generator: base.GeneratorLike):
        self._generator = generator
        for map_ in self._maps:
            map_(generator)


CustomMap = base.CustomMap


def map_(func):
    """Decorate an additional map step to the current pipeline.

    Note:
        This is for the decorator syntax for creating pipelines.

    Note:
        This uses `CustomMap`.

    Parameters
    ----------
    func : ``callable``
        The function to use for mapping.
    """
    return CustomMap(func)

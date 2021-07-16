"""`data.base.DataMap` subclasses to transform distributional data."""

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

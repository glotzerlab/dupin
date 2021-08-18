"""Base classes for the data module."""

import typing
from abc import abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, Dict, Union

import numpy as np
import numpy.typing

# must use strings to forward types
GeneratorLike = Union[
    "Generator",
    "DataMap",
    typing.Callable[..., Dict[str, Union[float, np.typing.ArrayLike]]],
]
GeneratorLike.__doc__ = """
A type hint for objects that act like data generators for event_detection.

The object can either be a `Generator`, `DataMap`, or callable with the
appropriate return value.
"""


def _join_filter_none(string, sequence):
    return string.join(filter(lambda x: x is not None, sequence))


class _DataModifier(Callable):
    """Generalized modifier of data in a pipeline."""

    def __init__(self, generator: GeneratorLike):
        """Create a DataReducer object.

        This is an abstract base class and cannot be instantiated directlty.

        Parameters
        ----------
        generator: GeneratorLike
            A generator like object to reduce.
        """
        self._generator = generator
        self._logger = None

    def __call__(self, *args: Any, **kwargs: Any):
        """Call the underlying generator performing the new modifications."""
        args, kwargs = self.update(args, kwargs)
        data = self._generator(*args, **kwargs)
        processed_data = {}
        for base_name, datum in data.items():
            if isinstance(datum, (Sequence, np.ndarray)):
                if self._logger is not None:
                    self._logger.set_context(base_name)
                processed_data.update(
                    {
                        _join_filter_none(
                            "_", (extension, base_name)
                        ): processed_datum
                        for extension, processed_datum in self.compute(
                            datum
                        ).items()
                    }
                )
            else:
                processed_data[base_name] = datum
        return processed_data

    @classmethod
    def wraps(cls, *args, **kwargs):
        """Create the class wrapping around the composed callable."""
        # Expects that generator is the first argument
        return lambda generator: cls(generator, *args, **kwargs)

    def update(cls, args, kwargs):
        """Update data modifier before compute if necessary.

        This is called before the internal generator is called. The method can
        consume arguments and returns the new args and kwargs (with potential
        arguments removed).
        """
        return args, kwargs

    @abstractmethod
    def compute(cls, distribution):
        """Perform the data modification on the array."""
        pass

    def attach_logger(self, logger):
        """Add a logger to this step in the data pipeline.

        Parameters
        ----------
        logger: event_detection.data.logging.Logger
            A logger object to store data from the data pipeline for individual
            elements of the composed maps.
        """
        self._logger = logger
        try:
            self._generator.attach_logger(logger)
        # Do nothing if generator does not have attach_logger logger function
        # (e.g. custom generator function).
        except AttributeError:
            pass

    def remove_logger(self):
        """Remove a logger from this step in the pipeline if it exists."""
        self._logger = None
        try:
            self._generator.remove_logger()
        # Do nothing if generator does not have remove_logger logger function
        # (e.g. custom generator function).
        except AttributeError:
            pass


class DataReducer(_DataModifier):
    """Base class for reducing distributions into scalar features.

    The class automatically skips over scalar features in its reduction.
    Subclasses requires the implemnation of `compute`.

    Note:
        This is an abstract base class and cannot be instantiated.
    """

    @abstractmethod
    def compute(self, distribution: np.typing.ArrayLike) -> Dict[str, float]:
        """Turn a distribution into scalar features.

        Parameters
        ----------
        distribution : :math:`(N,)` np.ndarray of float
            The array representing a distribution to reduce.

        Return
        -------
        reduced_distribution : dict[str, float]
            Returns a dictionary with string keys representing the type of
            reduction for its associated value. For instance if the value is the
            max of the distribution, a logical key value would be ``'max'``. The
            key only needs to represent the reduction, the original distribution
            name will be dealt automatically.
        """
        pass


class DataMap(_DataModifier):
    """Base class for mapping distributions to another distribution.

    When the raw distribution of a given simulation snapshot is not appropriate
    as a feature or requires further processing, a `DataMap` instance can be
    used to wrap a `Generator` instance for this processing. This class
    automatically skips over scalar features.

    This class requires the implemnation of `compute` in subclasses.

    Note:
        While this is named after the map operation, the array returned need not
        be identical in size.

    Note:
        This is an abstract base class and cannot be instantiated.
    """

    @abstractmethod
    def compute(self, data: np.typing.ArrayLike) -> np.typing.ArrayLike:
        """Turn a distribution into another distribution.

        Parameters
        ----------
        distribution : :math:`(N,)` np.ndarray of float
            The array representing a distribution to map.

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


class Generator(Callable):
    """The abstract base class for generating signals used for event detection.

    This just defines a simple interface through `__call__` where signals are
    generated with name pairs in a `dict`.
    """

    @abstractmethod
    def __call__(
        self, *args, **kwargs
    ) -> Dict[str, Union[float, np.typing.ArrayLike]]:
        """Return the output signal(s) for given inputs.

        This method can have an arbitrary signature in subclasses.

        Returns
        -------
        signals: dict[str, Union[float, numpy.typing.ArrayLike]]
            Returns a mapping of signal names to floating point or array like
            data. Array like data must be reduced before use in detection.
        """
        pass

    def attach_logger(self, logger):
        """Add a logger to this step in the data pipeline.

        Parameters
        ----------
        logger: event_detection.data.logging.Logger
            A logger object to store data from the data pipeline for individual
            elements of the composed maps.
        """
        self._logger = logger

    def remove_logger(self):
        """Remove a logger from this step in the pipeline if it exists."""
        self._logger = None

"""Base classes for the data module."""

import typing
from abc import abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt

# must use strings to forward types
GeneratorLike = Union[
    "Generator",
    "DataMap",
    typing.Callable[..., dict[str, Union[float, npt.ArrayLike]]],
]
GeneratorLike.__doc__ = """
A type hint for objects that act like data generators for dupin.

The object can either be a `Generator`, `DataMap`, or callable with the
appropriate return value.
"""


class PipeComponent:
    """Base class for piping methods for intermediate date pipeline elements.

    Provides helper methods for defining steps in a pipeline from a left to
    right or top to bottom approach.
    """

    def pipe(self, next_):
        """Add a step after current one in the data pipeline.

        Expects a `dupin.data.base.DataModifier` instance.

        Parameters
        ----------
        next_: dupin.data.base.DataModifier
            The next step in the data pipeline.

        Returns
        -------
        DataMap or DataReducer:
            Returns either a `DataMap` or `DataReducer` object based on the
            input to the method.
        """
        if isinstance(next_, DataModifier):
            return next_(self)
        if callable(next_):
            msg = (
                "To use custom callable use map or reduce as desired, "
                "or wrap in appropriate custom class."
            )
            raise ValueError(msg)
        msg = "Expected a DataModifier instance."
        raise ValueError(msg)

    def map(self, map_):
        """Add a mapping step after the current step in the data pipeline.

        Expects a custom callable or a `DataMap` instance.

        Parameters
        ----------
            map_: dupin.data.base.DataMap \
                    or ``callable`` [numpy.ndarray, dict[str, numpy.ndarray]]:
                The next step in the data pipeline. Can be a custom callable
                mapping function or an dupin any of the built in
                mapping operations.

        Returns
        -------
            DataMap:
                Returns either a `DataMap` subclass based on the passed in
                object.
        """
        if isinstance(map_, DataMap):
            return map_(self)
        if callable(map_):
            return CustomMap(self, map_)
        msg = "Expected a callable or a DataMap instance."
        raise ValueError(msg)

    def reduce(self, reduce_):
        """Add a reducing step after the current step in the data pipeline.

        Expects a custom callable or a `DataReducer` instance.

        Parameters
        ----------
            reduce_: dupin.data.base.DataReducer \
                    or ``callable`` [numpy.ndarray, dict[str, float]]
                The next step in the data pipeline. Can be a custom callable
                reducing function or an dupin any of the built in
                reducing operations.

        Returns
        -------
        DataReducer:
            Returns a `DataReducer` subclass based on the passed in object.
        """
        if isinstance(reduce_, DataReducer):
            return reduce_(self)
        if callable(reduce_):
            return CustomReducer(self, reduce_)
        msg = "Expected a callable or a DataReduce instance."
        raise ValueError(msg)


def _join_filter_none(
    string: str, sequence: typing.Sequence[Optional[str]]
) -> str:
    """Perform `str.join` except None's get skipped."""
    return string.join(s for s in sequence if s is not None)


class DataModifier(Callable):
    """Generalized modifier of data in a pipeline.

    This is an abstract base class and cannot be instantiated directlty.

    Parameters
    ----------
    generator: :py:obj:`~.GeneratorLike`
        A generator like object to modify.
    """

    def __init__(self):
        self._generator = None
        self._logger = None

    def __call__(self, *args: Any, **kwargs: Any):
        """Call the underlying generator performing the new modifications."""
        if self._generator is None:
            self._decorate(*args, **kwargs)
            return self
        args, kwargs = self.update(args, kwargs)
        data = self._generator(*args, **kwargs)
        processed_data = {}
        for base_name, datum in data.items():
            if isinstance(datum, (Sequence, np.ndarray)):
                if self._logger is not None:
                    self._logger._set_context(base_name)
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

    def update(self, args, kwargs):
        """Update data modifier before compute if necessary.

        This is called before the internal generator is called. The method can
        consume arguments and returns the new args and kwargs (with potential
        arguments removed).
        """
        return args, kwargs

    @abstractmethod
    def compute(self, distribution):
        """Perform the data modification on the array."""
        pass

    def attach_logger(self, logger):
        """Add a logger to this step in the data pipeline.

        Parameters
        ----------
        logger: dupin.data.logging.Logger
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

    def _decorate(self, generator: GeneratorLike):
        self._generator = generator


class DataReducer(DataModifier):
    """Base class for reducing distributions into scalar features.

    The class automatically skips over scalar features in its reduction.
    Subclasses requires the implemnation of `compute`.

    Note
    ----
        This is an abstract base class and cannot be instantiated.
    """

    @abstractmethod
    def compute(self, distribution: npt.ArrayLike) -> dict[str, float]:
        """Turn a distribution into scalar features.

        Parameters
        ----------
        distribution : :math:`(N,)` np.ndarray of float
            The array representing a distribution to reduce.

        Returns
        -------
        reduced_distribution : dict[str, float]
            Returns a dictionary with string keys representing the type of
            reduction for its associated value. For instance if the value is the
            max of the distribution, a logical key value would be ``'max'``. The
            key only needs to represent the reduction, the original distribution
            name will be dealt automatically.
        """
        pass


class DataMap(DataModifier, PipeComponent):
    """Base class for mapping distributions to another distribution.

    When the raw distribution of a given simulation snapshot is not appropriate
    as a feature or requires further processing, a `DataMap` instance can be
    used to wrap a `Generator` instance for this processing. This class
    automatically skips over scalar features.

    This class requires the implemnation of `compute` in subclasses.

    Note
    ----
        While this is named after the map operation, the array returned need not
        be identical in size.

    Note
    ----
        This is an abstract base class and cannot be instantiated.
    """

    @abstractmethod
    def compute(self, data: npt.ArrayLike) -> npt.ArrayLike:
        """Turn a distribution into another distribution.

        Parameters
        ----------
        distribution : :math:`(N,)` np.ndarray of float
            The array representing a distribution to map.

        Returns
        -------
        signals : dict[str, float]
            Returns a dictionary with string keys representing the type of
            reduction for its associated value. For instance if the value is the
            max of the distribution, a logical key value would be ``'max'``. The
            key only needs to represent the reduction, the original distribution
            name will be dealt with by a generator.
        """
        pass


class Generator(Callable, PipeComponent):
    """The abstract base class for generating signals used for event detection.

    This just defines a simple interface through `__call__` where signals are
    generated with name pairs in a `dict`.
    """

    @abstractmethod
    def __call__(
        self, *args, **kwargs
    ) -> dict[str, Union[float, npt.ArrayLike]]:
        """Return the output signal(s) for given inputs.

        This method can have an arbitrary signature in subclasses.

        Returns
        -------
        signals: dict[str, Union[float, numpy.ndarray]]
            Returns a mapping of signal names to floating point or array like
            data. Array like data must be reduced before use in detection.
        """
        pass

    def attach_logger(self, logger):
        """Add a logger to this step in the data pipeline.

        Parameters
        ----------
        logger: dupin.data.logging.Logger
            A logger object to store data from the data pipeline for individual
            elements of the composed maps.
        """
        self._logger = logger

    def remove_logger(self):
        """Remove a logger from this step in the pipeline if it exists."""
        self._logger = None


class CustomMap(DataMap):
    """Wrap a custom mapping callable.

    Parameters
    ----------
    custom_function : ``callable`` [`numpy.ndarray`, `dict` ]
        A custom callable that takes in a NumPy array and returns a dictionary
        with keys indicating the tranformation and values the transformed
        distribution (array).

    Attributes
    ----------
    function : ``callable`` [[`numpy.ndarray`], `dict` ]
        The provided callable.
    """

    def __init__(
        self,
        custom_function: typing.Callable[
            [npt.ArrayLike], dict[str, np.ndarray]
        ],
    ):
        super().__init__()
        self.function = custom_function

    def compute(self, data: npt.ArrayLike) -> npt.ArrayLike:
        """Call the internal function."""
        return self.function(data)


class CustomReducer(DataReducer):
    """Wrap a custom reducing callable.

    Parameters
    ----------
    custom_function: ``callable`` [`numpy.ndarray`, `dict` [`str`, `float` ]
        A custom callable that takes in a NumPy array and returns a
        dictionary with keys indicating the reduction and values the reduced
        distribution value.

    Attributes
    ----------
    function: ``callable`` [[`numpy.ndarray`], `dict` [`str`, `numpy.ndarray` ]]
        The provided callable.
    """

    def __init__(
        self,
        custom_function: typing.Callable[[npt.ArrayLike], dict[str, float]],
    ):
        super().__init__()
        self.function = custom_function

    def compute(self, data: npt.ArrayLike) -> npt.ArrayLike:
        """Call the internal function."""
        return self.function(data)


class CustomGenerator(Generator):
    """Wrap a user callable for starting a data pipeline.

    This class allows custom user functions to be the generator of the initial
    data from a pipeline. The call signature is arbitrary but has an expected
    output described in the parameter section.

    Parameters
    ----------
    custom_function: ``callable`` [[...], dict[str, numpy.ndarray or float]]
        A custom callable that returns a dictionary with feature names for
        keys and feature values for values (as either floats or arrays).

    Attributes
    ----------
    function: ``callable`` [[...], dict[str, numpy.ndarray or float]]
        The provided callable.
    """

    def __init__(self, custom_function):
        self.function = custom_function

    def __call__(self, *args, **kwargs):
        """Call the internal function."""
        return self.function(*args, **kwargs)


def make_generator(func):
    """Decorate an function to mark as a data generator.

    Note
    ----
        This is for the decorator syntax for creating pipelines.

    Note
    ----
        This uses `CustomGenerator`.

    Parameters
    ----------
    func : ``callable``
        The function to use for generating initial data.
    """
    return CustomGenerator(func)

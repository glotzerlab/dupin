"""Base classes for the data module."""

import typing
from abc import abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, Dict, Optional, Union

import numpy as np
import numpy.typing as npt

# must use strings to forward types
GeneratorLike = Union[
    "Generator",
    "DataMap",
    typing.Callable[..., Dict[str, Union[float, npt.ArrayLike]]],
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

        Expects the output of `dupin.data.base.DataModifier.wraps`.

        Parameters
        ----------
        next_: dupin.data.base.PreparedPipeComponent
            The next step in the data pipeline. To get such an object use
            `DataModifier.wraps`.

        Returns
        -------
        DataMap or DataReducer:
            Returns either a `DataMap` or `DataReducer` object based on the
            input to the method.
        """
        if isinstance(next_, PreparedPipeComponent):
            return next_(self)
        elif callable(next_):
            raise ValueError(
                "To use custom callable use map or reduce as desired, "
                "or wrap in appropriate custom class."
            )
        else:
            raise ValueError("Expected the output of DataModifier.wraps.")

    def map(self, map_):
        """Add a mapping step after the current step in the data pipeline.

        Expects a custom callable or the output of a `DataMap.wraps` call.

        Parameters
        ----------
            map_: dupin.data.base.PreparedPipeComponent \
                    or callable[numpy.ndarray, dict[str, numpy.ndarray]]:
                The next step in the data pipeline. Can be a custom callable
                mapping function or an ``dupin`` any of the built in
                mapping operations through `DataMap.wraps`.

        Returns
        -------
            DataMap:
                Returns either a `DataMap` subclass based on the passed in
                object.
        """
        if isinstance(map_, PreparedPipeComponent):
            if issubclass(map_._target_cls, DataMap):
                return map_(self)
            else:
                raise ValueError("Expected output of DataMap.wraps().")
        elif callable(map_):
            return CustomMap(self, map_)
        else:
            raise ValueError(
                "Expected a callable or the output of DataMap.wraps()"
            )

    def reduce(self, reduce_):
        """Add a reducing step after the current step in the data pipeline.

        Expects a custom callable or the output of a `DataReducer.wraps` call.

        Parameters
        ----------
            reduce_: dupin.data.base.PreparedPipeComponent \
                    or callable[numpy.ndarray, dict[str, float]]
                The next step in the data pipeline. Can be a custom callable
                reducing function or an ``dupin`` any of the built in
                reducing operations through `DataReducer.wraps`.

        Returns
        -------
        DataReducer:
            Returns a `DataReducer` subclass based on the passed in object.
        """
        if isinstance(reduce_, PreparedPipeComponent):
            if issubclass(reduce_._target_cls, DataReducer):
                return reduce_(self)
            else:
                raise ValueError("Expected output of DataReduce.wraps().")
        elif callable(reduce_):
            return CustomReducer(self, reduce_)
        else:
            raise ValueError(
                "Expected a callable or the output of DataReduce.wraps()"
            )


class PreparedPipeComponent:
    """An intermediate class allowing for piping and decorating data pipelines.

    This class is the output of `DataModifier.wraps` and allows for both the
    pipeline and decorator syntax for creating data pipelines.

    Warning:
        The class should not be instantiated directly.
    """

    def __init__(self, cls, *args, **kwargs):
        """Create a PreparedPipeComponent object."""
        self._target_cls = cls
        self._args = args
        self._kwargs = kwargs

    def __call__(self, generator):
        """Create internal pipeline component."""
        return self._target_cls(generator, *self._args, **self._kwargs)


def _join_filter_none(
    string: str, sequence: typing.Sequence[Optional[str]]
) -> str:
    """Perform `str.join` except None's get skipped."""
    return string.join(s for s in sequence if s is not None)


class DataModifier(Callable):
    """Generalized modifier of data in a pipeline."""

    def __init__(self, generator: GeneratorLike):
        """Create a DataModifier object.

        This is an abstract base class and cannot be instantiated directlty.

        Parameters
        ----------
        generator: :py:obj:`~.GeneratorLike`
            A generator like object to modify.
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
        return PreparedPipeComponent(cls, *args, **kwargs)

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


class DataReducer(DataModifier):
    """Base class for reducing distributions into scalar features.

    The class automatically skips over scalar features in its reduction.
    Subclasses requires the implemnation of `compute`.

    Note:
        This is an abstract base class and cannot be instantiated.
    """

    @abstractmethod
    def compute(self, distribution: npt.ArrayLike) -> Dict[str, float]:
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


class DataMap(DataModifier, PipeComponent):
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
    def compute(self, data: npt.ArrayLike) -> npt.ArrayLike:
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


class Generator(Callable, PipeComponent):
    """The abstract base class for generating signals used for event detection.

    This just defines a simple interface through `__call__` where signals are
    generated with name pairs in a `dict`.
    """

    @abstractmethod
    def __call__(
        self, *args, **kwargs
    ) -> Dict[str, Union[float, npt.ArrayLike]]:
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

    Attributes
    ----------
    function: ``callable`` [[`numpy.ndarray`], `dict` [`str`, `numpy.ndarray`]]
        The provided callable.
    """

    def __init__(
        self,
        generator: GeneratorLike,
        custom_function: typing.Callable[
            [npt.ArrayLike], Dict[str, np.ndarray]
        ],
    ):
        """Create a `CustomMap` object.

        Parameters
        ----------
        generator: :py:obj:`~.GeneratorLike`
            A generator like object to transform.
        custom_function: ``callable`` [`numpy.ndarray`, \
                `dict` [`str`, `numpy.ndarray`]
            A custom callable that takes in a NumPy array and returns a
            dictionary with keys indicating the tranformation and values the
            transformed distribution (array).
        """
        super().__init__(generator)
        self.function = custom_function

    def compute(self, data: npt.ArrayLike) -> npt.ArrayLike:
        """Call the internal function."""
        return self.function(data)


class CustomReducer(DataReducer):
    """Wrap a custom reducing callable.

    Attributes
    ----------
    function: ``callable`` [[`numpy.ndarray`], `dict` [`str`, `numpy.ndarray`]]
        The provided callable.
    """

    def __init__(
        self,
        generator: GeneratorLike,
        custom_function: typing.Callable[[npt.ArrayLike], Dict[str, float]],
    ):
        """Create a `CustomReducer` object.

        Parameters
        ----------
        generator: :py:obj:`~.GeneratorLike`
            A generator like object to reduce.
        custom_function: ``callable`` [`numpy.ndarray`, `dict` [`str`, `float` ]
            A custom callable that takes in a NumPy array and returns a
            dictionary with keys indicating the reduction and values the reduced
            distribution value.
        """
        super().__init__(generator)
        self.function = custom_function

    def compute(self, data: npt.ArrayLike) -> npt.ArrayLike:
        """Call the internal function."""
        return self.function(data)


class CustomGenerator(Generator):
    """Wrap a user callable for starting a data pipeline.

    This class allows custom user functions to be the generator of the initial
    data from a pipeline. The call signature is arbitrary but has an expected
    output given in the `__init__` documentation.

    Attributes
    ----------
    function: callable[[...], dict[str, numpy.ndarray or float]]
        The provided callable.
    """

    def __init__(self, custom_function):
        """Wrap a user callable for using a data pipeline.

        Parameters
        ----------
        custom_function: callable[[...], dict[str, numpy.ndarray or float]]
            A custom callable that returns a dictionary with feature names for
            keys and feature values for values (as either floats or arrays).
        """
        self.function = custom_function

    def __call__(self, *args, **kwargs):
        """Call the internal function."""
        return self.function(*args, **kwargs)


def make_generator(func):
    """Decorate an function to mark as a data generator.

    Note:
        This uses `CustomGenerator`.

    Parameters
    ----------
    func : callable
        The function to use for generating initial data.
    """
    return CustomGenerator(func)

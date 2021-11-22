"""Helper module for getting features accross an entire trajectory."""


from typing import Any, Dict, Iterator, Optional, Tuple

import dupin.errors as errors

from . import base, logging

try:
    import pandas as pd
except ImportError:
    pd = errors._RaiseModuleError("pandas")


class SignalAggregator:
    """Using generators computes signals across a trajectory.

    This class can be used to create appropriate data structures for use in
    analysising a whole trajectory with offline methods or iteratively
    analysising for online use. See the `compute` and `accumulate` methods for
    usage.
    """

    def __init__(
        self,
        generator: base.GeneratorLike,
        logger: Optional[logging.Logger] = None,
    ):
        """Create a `SignalAggregator` object.

        Parameters
        ----------
        generator: generator_like
            A sequence of signal generators to use for generating the
            multivariate signal of a trajectory.
        logger: dupin.data.logging.Logger or None
            A logger object to store information about the data processing of
            the given pipeline. Defaults to None
        """
        self.generator = generator
        self.signals = []
        self.logger = logger

    def compute(
        self, iterator: Iterator[Tuple[Tuple[Any, ...], Dict[str, Any]]]
    ):
        """Compute signals from generator across the iterator.

        These signals are stored internally until asked for by `to_dataframe`.
        This can be called multiple times, and the stored signals values will be
        appended.

        Parameters
        ----------
        iterator: Iterator[Tuple[Tuple[Any,...], Dict[str, Any]]]
            An object when iterated over, yields args and kwargs compatible with
            the ``generator_like`` object's call signature.

        Note:
            Use the `from_base_iterator` staticmethod to convert a standard
            iterator into one compatible with this method.
        """
        for args, kwargs in iterator:
            self.accumulate(*args, **kwargs)

    def accumulate(self, *args: Any, **kwargs: Any):
        r"""Add features from simulation snapshot to object.

        Allows the addition of individual snapshots to aggregator. This can be
        useful for online detection or any case where computing the entire
        trajectory is not possible or not desired.

        Parameters
        ----------
        \*args: Any
            Positional arguments to feed to the generator like object.
        \*\*kwargs: Any
            Keyword arguments to feed to the generator like object.
        """
        self.signals.append(self.generator(*args, **kwargs))
        if self._logger is not None:
            self._logger.end_frame()

    def to_dataframe(self) -> "pd.DataFrame":
        """Return the aggregated signals as a pandas DataFrame.

        Note:
            This method requires pandas to be available.

        Returns
        -------
        signals: pandas.DataFrame
            The aggregated signals. The columns are features, and the indices
            correspond to system frames in the order passed to `accumulate` or
            `compute`.
        """
        return pd.DataFrame(
            {
                col: [frame[col] for frame in self.signals]
                for col in self.signals[0]
            }
        )

    @staticmethod
    def from_base_iterator(
        iterator: Iterator[Any], is_args: bool = False, is_kwargs: bool = False
    ):
        """Convert a base iterator into one that works with `compute`.

        The default behavior is to treat the items of the iterator as a single
        positional argument. Read the argument list for alternative options.

        Parameters
        ----------
        iterator: Iterator[Any]
            The iterator to convert.
        is_args: bool, optional
            Whether to treat the iterator objects as positional arguments.
            Defaults to False.
        is_kwargs: bool, optional
            Whether to treat the iterator objects as keyword arguments. Defaults
            to False.

        Returns
        -------
        new_iterator: Iterator[Tuple[Tuple[Any,...], Dict[str, Any]]]
            The modified iterator.
        """
        return (((arg,), {}) for arg in iterator)

    @property
    def logger(self):
        """dupin.data.logging.Logger: Logger for the aggregator."""
        return self._logger

    @logger.setter
    def logger(self, new_logger):
        self._logger = new_logger
        if new_logger is None:
            try:
                self.generator.remove_logger()
            except AttributeError:
                pass
        try:
            self.generator.attach_logger(new_logger)
        except AttributeError:
            pass

"""Helper module for generating/storing features accross an entire trajectory.

This class provides the `SignalAggregator` class which takes a pipeline and
provides methods for storing the output across a trajectory.
"""


from collections.abc import Iterator
from typing import Any, Optional

import numpy as np

from .. import errors
from . import base, logging

try:
    import pandas as pd
except ImportError:
    pd = errors._RaiseModuleError("pandas")
try:
    import xarray as xa
except ImportError:
    xa = errors._RaiseModuleError("xarray")


class SignalAggregator:
    """Using generators computes signals across a trajectory.

    This class can be used to create appropriate data structures for use in
    analyzing a whole trajectory with offline methods or iteratively
    analyzing for online use. See the `compute` and `accumulate` methods for
    usage.

    Parameters
    ----------
    generator : :obj:`dupin.data.base.GeneratorLike`
        A sequence of signal generators to use for generating the
        multivariate signal of a trajectory.
    logger : dupin.data.logging.Logger
        A logger object to store information about the data processing of
        the given pipeline. Defaults to ``None``.

    Attributes
    ----------
    generator : :obj:`dupin.data.base.GeneratorLike`
        The generator which generates data given a trajectory frame.
    signals : list[dict]
        The current list of analyzed frames.
    """

    def __init__(
        self,
        generator: base.GeneratorLike,
        logger: Optional[logging.Logger] = None,
    ):
        self.generator = generator
        self.signals = []
        self.logger = logger

    def compute(
        self, iterator: Iterator[tuple[tuple[Any, ...], dict[str, Any]]]
    ):
        """Compute signals from generator across the iterator.

        These signals are stored internally in ``signals`` until asked for by
        `to_dataframe` or `to_xarray`. This can be called multiple times, and
        the stored signals values will be appended.

        Parameters
        ----------
        iterator: Iterator[Tuple[Tuple[Any,...], Dict[str, Any]]]
            An object when iterated over, yields args and kwargs compatible with
            the ``generator_like`` object's call signature.

        Note
        ----
            Use the `from_base_iterator` staticmethod to convert a standard
            iterator into one compatible with this method.
        """
        for args, kwargs in iterator:
            self.accumulate(*args, **kwargs)

    def accumulate(self, *args: Any, **kwargs: Any):
        r"""Add features from simulation snapshot to object.

        Allows the addition of individual snapshots to aggregator. This can be
        useful for online detection or any case where computing the entire
        trajectory is not possible or not desired or for use in a for loop.

        Parameters
        ----------
        \*args:
            Positional arguments to feed to the generator like object.
        \*\*kwargs:
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
        if len(self.signals) == 0:
            return pd.DataFrame()

        def is_array(v):
            if hasattr(v, "__len__"):
                return len(v) > 1
            return False

        # We assume that all frames "look" alike in their structure. If this is
        # broken than this check will not catch it. However, this breaks one our
        # assumptions so is outside what we should check for.
        if any(is_array(value) for value in self.signals[0].values()):
            msg = (
                "Signal is 3 dimensional. ~.to_dataframe requires a 2 "
                "dimensional signal. Use ~.to_xarray instead."
            )
            raise ValueError(msg)
        return pd.DataFrame(
            {
                col: [frame[col] for frame in self.signals]
                for col in self.signals[0]
            }
        )

    def to_xarray(self, third_dim_name="third_dim") -> "xa.DataArray":
        """Return the aggregated signal as a `xarray.DataArray`.

        This method is designed to be used primarily with non-reduced data (e.g.
        per-particle features). This enables with `XarrayGenerator` to do the
        mapping/reduction later, attempt multiple reductions, or use the data
        for purposes outside detection such as visualization or plotting.

        Note:
            This method requires `xarray` to be available.

        Warning:
            This method only works when all arrays have the same first dimension
            size.

        Returns
        -------
        signal: xarray.DataArray
            The aggregated signal. The first dimension is frames, the second is
            features, and the third the first dimension of the aggregated
            features (eg. number of particles).
        """
        feature_order = self.signals[0].keys()
        data = np.stack(
            [
                np.stack([frame[k].squeeze() for k in feature_order])
                for frame in self.signals
            ]
        )
        return xa.DataArray(
            data=data,
            dims=("frame", "feature", third_dim_name),
            coords={"feature": list(feature_order)},
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
        iterator : Iterator[Any]
            The iterator to convert.
        is_args : :obj:`bool`, optional
            Whether to treat the iterator objects as positional arguments (i.e.
            yields tuples). Defaults to False.
        is_kwargs : :obj:`bool`, optional
            Whether to treat the iterator objects as keyword arguments (i.e.
            yields dicts). Defaults to False.

        Returns
        -------
        new_iterator: Iterator[Tuple[Tuple[Any,...], Dict[str, Any]]]
            The modified iterator.
        """
        if is_args:
            return ((arg, {}) for arg in iterator)
        if is_kwargs:
            return (((), arg) for arg in iterator)
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


class XarrayGenerator(base.Generator):
    """Generator that converts a frame from xarray to a dupin compatible form.

    This class is useful to use with `SignalAggregator.to_xarray` to separate
    the data generation and optionally the mapping step from the reduction step.

    Parameters
    ----------
    feature_dim : :obj:`str`, optional
        The name of the feature dimension in the xarray frames, defaults to
        "feature" (the default of `SignalAggregator.to_xarray`).
    """

    def __init__(self, feature_dim: str = "feature"):
        self._feature_dim = feature_dim

    def __call__(self, xarray_frame: "xa.DataArray"):
        """Convert the xarray object to a dupin pipeline representation.

        Parameters
        ----------
        xarray_frame : xarray.DataArray
            The data array for the current frame of the signal.

        Returns
        -------
        frame : dict [str, numpy.ndarray]
            The data represented as a dictionary of arrays.
        """
        return {
            feature: xarray_frame.sel({self._feature_dim: feature}).to_numpy()
            for feature in map(
                str, xarray_frame.coords[self._feature_dim].to_numpy()
            )
        }


__all__ = ["SignalAggregator", "XarrayGenerator"]

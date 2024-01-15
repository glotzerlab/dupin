"""Classes for transforming array quantities into scalar features.

Reduction in ``dupin`` takes an array and _reduces_ it to a set number of scalar
values. A computer science reduction goes from an array to a single value. Our
usage of the term is similar; we just allow for multiple reductions to happen
within the same reducer. Examples of common reducers in the ``dupin`` sense are
the max, min, mean, mode, and standard deviation functions.
"""

import warnings
from typing import Optional

import numpy as np
import numpy.typing as npt

from . import base


class Percentile(base.DataReducer):
    """Reduce a distribution into percentile values.

    The reducers sorts the input array to get the provided percentiles. The
    reducers then uses the key format f"{percentile}%" to identify it
    reductions.

    Parameters
    ----------
    percentiles : `tuple` [ `int` ], optional
        The percentiles in integer form (i.e. 100% equals 100). By defualt,
        every 10% increment from 0% to 100% (inclusive) is taken.
    """

    def __init__(self, percentiles: Optional[tuple[int]] = None) -> None:
        if percentiles is None:
            percentiles = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
        if len(percentiles) == 0:
            msg = "Cannot have an empty percentiles sequence."
            raise ValueError(msg)
        self._percentiles = np.unique(list(percentiles))
        self._quantiles = self._percentiles / 100.0
        super().__init__()

    def compute(self, distribution: np.ndarray) -> dict[str, float]:
        """Return the reduced distribution."""
        if len(distribution) == 0:
            return {}

        data = {}
        log = {}
        non_nan = np.flatnonzero(~np.isnan(distribution))
        if non_nan.size == 0:
            for p in self._percentiles:
                key = f"{p}%"
                log[key] = 0
                data[key] = np.nan
            if self._logger is not None and len(log) > 0:
                self._logger["Percentile"] = log
            return data

        cleaned_dist = distribution[non_nan]
        indices = self._get_indices(len(cleaned_dist))
        sorted_indices = np.argsort(cleaned_dist)
        for i, p in zip(indices, self._percentiles):
            key = f"{p}%"
            log[key] = non_nan[sorted_indices[i]]
            data[key] = cleaned_dist[sorted_indices[i]]
        if self._logger is not None and len(log) > 0:
            self._logger["Percentile"] = log
        return data

    def _get_indices(self, distribution_size):
        """Map to percentiles to [0, len - 1]."""
        fractional_indices = self._quantiles * (distribution_size - 1)
        return np.rint(fractional_indices).astype(int)


class NthGreatest(base.DataReducer):
    """Reduce a distribution to the Nth greatest values.

    This reducer returns the greatest and least values from a distribution as
    specified by the provided indices. Greatest values are specified by positive
    integers and least by negative, e.g. -1 is the minimum value in the array.
    The features keys are modified with the index ordinal number and whether it
    is greatest or least. -1 becomes "1st_least" and 10 becomes "10th_greatest".

    Parameters
    ----------
    indices : `list` [ `int` ], optional
        The values to query. 1 is the greatest value in the distribution; 10
        the tenth, and so on. Negative number consitute the smallest values
        in the distribution. -1 is the least value in the distribution. 0 is
        treated as 1.
    """

    def __init__(self, indices: tuple[int]) -> None:
        if len(indices) == 0:
            msg = "Cannot have an empty indices sequence."
            raise ValueError(msg)
        self._indices = self._fix_indices(np.asarray(list(indices)))
        self._names = [self._index_name(index) for index in self._indices]
        super().__init__()

    def compute(self, distribution: np.ndarray) -> dict[str, float]:
        """Return the signals with modified keys."""
        if len(distribution) == 0:
            warnings.warn("Received empty array.", stacklevel=2)
            return {}
        nan_mask = np.flatnonzero(~np.isnan(distribution))
        filtered_distribution = distribution[nan_mask]
        sorted_indices = np.argsort(filtered_distribution)
        log = {}
        data = {}
        for i, name in zip(self._indices, self._names):
            set_to_nan = False
            if not self._fits(distribution, i):
                set_to_nan = True
                warnings.warn(
                    "Not enough elements found for NthGreatest, setting to "
                    "nan.",
                    stacklevel=2,
                )

            if not self._fits(filtered_distribution, i):
                set_to_nan = True
                warnings.warn(
                    "Not enough non-nan elements found for NthGreatest, "
                    "setting to nan.",
                    stacklevel=2,
                )
            if set_to_nan:
                data[name] = np.nan
                log[name] = np.nan
                continue
            data[name] = filtered_distribution[sorted_indices[i]]
            log[name] = nan_mask[sorted_indices[i]]
        if self._logger is not None and len(log) > 0:
            self._logger["NthGreatest"] = log
        return data

    @staticmethod
    def _fits(dist: np.array, index: int):
        if index < 0:
            return -index <= len(dist)
        return index < len(dist)

    @staticmethod
    def _index_name(index: int) -> str:
        if index >= 0:
            index += 1
        type_ = "least" if index > 0 else "greatest"
        abs_index = abs(index)
        unit_value = abs_index % 10
        # add appropriate suffix
        if unit_value == 1:
            suffix = "st"
        elif unit_value == 2:  # noqa: PLR2004
            suffix = "nd"
        elif unit_value == 3:  # noqa: PLR2004
            suffix = "rd"
        else:
            suffix = "th"
        return f"{abs_index}{suffix}_{type_}"

    @staticmethod
    def _fix_indices(indices: list[int]) -> list[int]:
        neg_indices = -indices
        return np.unique(np.where(indices > 0, neg_indices, neg_indices - 1))


class Tee(base.DataReducer):
    """Enable mutliple reducers to act on the same generator like object.

    Each reducer is run on the original distribution and their reductions are
    concatenated. This reducer does not create its own reductions or
    corresponding keys.

    Parameters
    ----------
    reducers: `list` [`dupin.data.base.DataReducer`]
        A sequence of a data reducers.
    """

    def __init__(
        self,
        reducers: list[base.DataReducer],
    ):
        if len(reducers) == 0:
            msg = "Cannot have empty reducers sequence."
            raise ValueError(msg)
        self._reducers = reducers
        super().__init__()

    def compute(self, distribution: npt.ArrayLike) -> dict[str, float]:
        """Run all composed reducer computes."""
        processed_data = {}
        for reducer in self._reducers:
            processed_data.update(reducer.compute(distribution))
        return processed_data

    def attach_logger(self, logger):
        """Add a logger to this step in the data pipeline.

        Parameters
        ----------
        logger: dupin.data.logging.Logger
            A logger object to store data from the data pipeline for individual
            elements of the composed maps.
        """
        self._logger = logger
        for reducer in self._reducers:
            try:
                reducer.attach_logger(logger)
            # Do nothing if generator does not have attach_logger logger
            # function (e.g. custom map function).
            except AttributeError:
                pass

    def remove_logger(self):
        """Remove a logger from this step in the pipeline if it exists."""
        self._logger = None
        for reducer in self._reducers:
            try:
                reducer.remove_logger()
            # Do nothing if generator does not have remove_logger logger
            # function (e.g. custom reducer function).
            except AttributeError:
                pass

    def _decorate(self, generator: base.GeneratorLike):
        self._generator = generator
        for reducer in self._reducers:
            reducer._decorate(generator)


CustomReducer = base.CustomReducer


def reduce_(func):
    """Add the reduce step to the current pipeline.

    Note:
        This is for the decorator syntax for creating pipelines.

    Note:
        This uses `CustomReducer`.

    Parameters
    ----------
    func : ``callable``
        The function to use for reducing.
    """
    return CustomReducer(func)

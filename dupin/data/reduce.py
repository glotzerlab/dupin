"""Classes for transforming array quantities into scalar features.

Reduction in ``dupin`` takes an array and _reduces_ it to a set number of scalar
values. A computer science reduction goes from an array to a single value. Our
usage of the term is similar; we just allow for multiple reductions to happen
within the same reducer. Examples of common reducers in the ``dupin`` sense are
the max, min, mean, mode, and standard deviation functions.
"""

from typing import Dict, List, Optional, Tuple

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

    def __init__(self, percentiles: Optional[Tuple[int]] = None) -> None:
        if percentiles is None:
            percentiles = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
        self._percentiles = percentiles
        super().__init__()

    def compute(self, distribution: np.ndarray) -> Dict[str, float]:
        """Return the reduced distribution."""
        indices = self._get_indices(len(distribution))
        sorted_indices = np.argsort(distribution)
        if self._logger is not None:
            self._logger["Percentile"] = {
                f"{percent}%": sorted_indices[index]
                for index, percent in zip(indices, self._percentiles)
            }
        return {
            f"{percent}%": distribution[sorted_indices[index]]
            for percent, index in zip(self._percentiles, indices)
        }

    def _get_indices(self, distribution_size):
        """Map to percentiles to [0, len - 1]."""
        return np.round(
            np.array(self._percentiles) / 100 * (distribution_size - 1),
            decimals=0,
        ).astype(int)


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

    def __init__(self, indices: Tuple[int]) -> None:
        self._indices = self._fix_indices(indices)
        self._names = [self._index_name(index) for index in self._indices]
        super().__init__()

    def compute(self, distribution: np.ndarray) -> Dict[str, float]:
        """Return the signals with modified keys."""
        sorted_indices = np.argsort(distribution)
        if self._logger is not None:
            self._logger["NthGreatest"] = {
                name: sorted_indices[index]
                for index, name in zip(self._indices, self._names)
            }
        return {
            name: distribution[sorted_indices[index]]
            for index, name in zip(self._indices, self._names)
        }

    @staticmethod
    def _index_name(index: int) -> str:
        if index >= 0:
            index += 1
        type_ = "least" if index > 0 else "greatest"
        abs_index = abs(index)
        unit_value = abs_index % 10
        if unit_value == 1:
            suffix = "st"
        elif unit_value == 2:
            suffix = "nd"
        elif unit_value == 3:
            suffix = "rd"
        else:
            suffix = "th"
        return f"{abs_index}{suffix}_{type_}"

    @staticmethod
    def _fix_indices(indices: List[int]) -> List[int]:
        array_indices = np.asarray(indices)
        neg_array_indices = -array_indices
        return np.unique(
            np.where(
                array_indices > 0, neg_array_indices, neg_array_indices - 1
            )
        )


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
        reducers: List[base.DataReducer],
    ):
        self._reducers = reducers
        super().__init__()

    def compute(self, distribution: npt.ArrayLike) -> Dict[str, float]:
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
            reducer(generator)


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

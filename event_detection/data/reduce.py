"""Classes for transforming array quantities into scalar features."""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing

from . import base


class Percentile(base.DataReducer):
    """Reduce a distribution into percentile values."""

    def __init__(
        self,
        generator: base.GeneratorLike,
        percentiles: Optional[Tuple[int]] = None,
    ) -> None:
        """Create a `Percentile` object.

        Parameters
        ----------
        generator: generator_like
            A generator like object to reduce to percentiles.

        percentiles : tuple[int], optional
            The percentiles in integer form (i.e. 100% equals 100). By defualt,
            every 10% increment from 0% to 100% (inclusive) is taken.
        """
        if percentiles is None:
            percentiles = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
        self._percentiles = percentiles
        super().__init__(generator)

    def compute(self, distribution: np.ndarray) -> Dict[str, float]:
        """Return the signals with dd% keys."""
        return {
            f"{percent}%": value
            for percent, value in zip(
                self._percentiles,
                np.percentile(distribution, self._percentiles),
            )
        }


class NthGreatest(base.DataReducer):
    """Reduce a distribution to the Nth greatest values.

    This reducer returns the greatest and least values from a distribution as
    specified by the provided indices. Greatest values are specified by positive
    integers and least by negative. The features keys are modified with the
    index ordinal number and whether it is greatest or least.
    """

    def __init__(
        self, generator: base.GeneratorLike, indices: Tuple[int]
    ) -> None:
        """Create a `NthGreatest` object.

        Parameters
        ----------
        generator: generator_like
            A generator like object to reduce to specified indices.
        indices : list[int], optional
            The values to query. 1 is the greatest value in the distribution; 10
            the tenth, and so on. Negative number consitute the smallest values
            in the distribution. -1 is the least value in the distribution. 0 is
            treated as 1.
        """
        self._indices = self._fix_indices(indices)
        self._names = [self._index_name(index) for index in self._indices]
        super().__init__(generator)

    def compute(self, distribution: np.ndarray) -> Dict[str, float]:
        """Return the signals with modified keys."""
        sorted_array = np.sort(distribution)
        return {
            name: sorted_array[-index]
            for index, name in zip(self._indices, self._names)
        }

    @staticmethod
    def _index_name(index: int) -> str:
        if index >= 0:
            index += 1
        type_ = "greatest" if index > 0 else "least"
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
    """Enable mutliple reducers to act on the same generator like object."""

    def __init__(
        self,
        generator: base.GeneratorLike,
        reducers: List[Callable[[base.GeneratorLike], base.DataReducer]],
    ):
        """Create a data.reduce.Tee object.

        Parameters
        ----------
        generator: generator_like
            A generator like object to reduce.
        reducers: list[callable[base.GeneratorLike, base.DataReducer]]
            A sequence of callables that take a generator like object and
            returns a data reducer. Using the ``wraps`` class method with a
            `DataReducer` subclass is a useful combination.
        """
        self._reducers = [reduce(generator) for reduce in reducers]
        super().__init__(generator)

    def compute(self, distribution: np.typing.ArrayLike) -> Dict[str, float]:
        """Run all composed reducer computes."""
        processed_data = {}
        for reducer in self._reducers:
            processed_data.update(reducer.compute(distribution))
        return processed_data

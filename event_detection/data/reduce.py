"""Classes for transforming array quantities into scalar features."""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

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
        generator: GeneratorLike
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
    """Reduce a distribution to the Nth greatest values."""

    def __init__(
        self, generator: base.GeneratorLike, indices: Tuple[int]
    ) -> None:
        """Create a `NthGreatest` object.

        Parameters
        ----------
        generator: GeneratorLike
            A generator like object to reduce to specified indices.
        indices : tuple[int], optional
            The indices to query. Negative indices are the Nth smallest values.
            Zero is not smallest value in the array.
        """
        self._indices = indices
        super().__init__(generator)

    def compute(self, distribution: np.ndarray) -> Dict[str, float]:
        """Return the signals with modified keys."""
        sorted_array = np.sort(distribution)
        return {
            f"{index}-th-{'greatest' if index > 0 else 'least'}": sorted_array[
                -index
            ]
            for index in self._indices
        }


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
        generator: GeneratorLike
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

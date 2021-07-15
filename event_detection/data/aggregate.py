"""Helper module for getting features accross an entire trajectory."""


from typing import List

import pandas as pd

from event_detection import signal


class SignalAggregator:
    """Using generators computes signals across a trajectory.

    This class can be used to create appropriate data structures for use in
    analysising a whole trajectory with offline methods or iteratively
    analysising for online use. See the `compute` and `accumulate` methods for
    usage.
    """

    def __init__(self, generators: List[signal.Generator]):
        """Create a `SignalAggregator` object.

        Parameters
        ----------
        generators: list[event_detection.signal.Generator]
            A sequence of signal generators to use for generating the
            multivariate signal of a trajectory.
        """
        self._generators = generators
        self._signals = []

    def compute(self, trajectory):
        """Compute signals from generators across the trajectory.

        These signals are stored internally unto asked for by `to_dataframe`.
        This can be called multiple times, and the stored signals values will be
        appended.

        Parameters
        ----------
        trajectory: a trajectory-like object
            An object when iterated over, yields objects compatible with
            `signal.Generator` objects. Examples include `gsd.hoomd.Trajectory`
            and a Python generator of ``(box, positions)`` tuples.
        """
        for system in trajectory:
            self.accumulate(system)

    def accumulate(self, system):
        """Add features from simulation snapshot to object.

        Allows the addition of individual snapshots to aggregator. This can be
        useful for online detection or any case where computing the entire
        trajectory is not possible or not desired.

        Parameters
        ----------
        system: a system-like object
            An object compatible with the current generators.
        """
        self._signals.append(
            {
                name: signal
                for generator in self._generators
                for name, signal in generator.generate(system).items()
            }
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Return the aggregated signals as a pandas DataFrame.

        Returns
        -------
        signals: pandas.DataFrame
            The aggregated signals. The columns are features, and the indices
            correspond to system frames in the order passed to `accumulate` or
            `compute`.
        """
        return pd.DataFrame(
            {
                col: [frame[col] for frame in self._signals]
                for col in self._signals[0]
            }
        )

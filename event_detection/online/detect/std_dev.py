"""Use the standard deviation of signals from a baseline to detect events."""

from copy import copy
from typing import Dict, List, Tuple, Union

import numpy as np

from event_detection.signal import Generator

from . import Detector, DetectorStatus


class _SignalStats:
    def __init__(self) -> None:
        self._baseline = []
        self.active_counter = 0
        self._recompute_stats = True

    def update_baseline(self, entry: float) -> None:
        self._baseline.append(entry)
        self._recompute_stats = True

    def compute(self):
        baseline = np.array(self._baseline)
        self._mean = np.mean(baseline)
        self._std_dev = np.std(baseline)

    def __getitem__(self, index: Union[int, slice]):
        return copy(self._baseline[index])

    @property
    def mean(self) -> float:
        if self._recompute_stats:
            self.compute()
        return self._mean

    @property
    def std_dev(self) -> float:
        if self._recompute_stats:
            self.compute()
        return self._std_dev


class StdDevDetector(Detector):
    """Detects events using the standard deviation of a signal from a baseline.

    For array signals percentiles are taken and used as individual distributions
    to measure the baseline for.

    Attributes
    ----------
    n_training: int
        The number of calls to use for generating the baseline before attempting
        to detect a signal.
    n_stdev: float
        The number of standard deviations from the base line that consitutes
        triggering an event.
    status: DetectorStatus
        The current status of the detector.
    """

    def __init__(
        self,
        generators: List[Generator],
        n_training: int = 50,
        n_std_dev: float = 2.0,
        threshold: int = 10,
    ):
        """Create a `StdDevDetector` object.

        Parameters
        ----------
        generators: list[Generator]
            The generators used for signal generation to use for event
            detection.
        n_training: int, optional
            The number of calls to use for generating the baseline before
            attempting to detect a signal.
        n_stdev: float, optional
            The number of standard deviations from the base line that consitutes
            triggering an event.
        theshold: int, optional
            The number of consequative calls that a signal is over/under the set
            number of standard deviations.
        """
        self._generators = generators
        self.n_training = n_training
        self.n_std_dev = n_std_dev
        self._signals_stats = {}
        self._count = 0
        self._threshold = threshold
        self._events = []
        self.status = DetectorStatus.INACTIVE

    def _get_signals(self, state) -> Dict[str, float]:
        signals = {}
        for generator in self.generators:
            new_signals = generator.generate(state)
            if signals.keys().isdisjoint(new_signals):
                signals.update(new_signals)
            else:
                raise RuntimeError("All signals must have unique names.")
        return signals

    def _update_stats(self, signals: Dict[str, float]) -> None:
        for signal_name, signal in signals.items():
            self._signals_stats[signal_name].update_baseline(signal)

    def _initialize_stats(self, signals: Dict[str, float]) -> None:
        for signal_name, signal in signals.items():
            signal_stats = _SignalStats()
            signal_stats.update_baseline(signal)
            self._signals_stats[signal_name] = signal_stats

    def _evaluate_signals(self, signals: Dict[str, float]):
        max_event_counter = 0
        confirmed_signals = []
        for signal_name, signal in signals.items():
            signal_stats = self._signals_stats[signal_name]
            deviation = abs(signal_stats.mean - signal)
            if deviation / signal_stats.std_dev > self.n_std_dev:
                signal_stats.active_counter += 1
                if signal_stats.active_counter > max_event_counter:
                    max_event_counter = signal_stats.active_counter
                    confirmed_signals.append(signal_name)
            else:
                signal_stats.active_counter = 0

        if max_event_counter > self._threshold:
            self.status = DetectorStatus.CONFIRMED
            self._events.append(self._count, confirmed_signals)
        elif max_event_counter > 0:
            self.status = DetectorStatus.ACTIVE
        else:
            self.status = DetectorStatus.INACTIVE
        return self.status

    def _update_status(self, state) -> DetectorStatus:
        signals = self._get_signals(state)
        if self._count == 0:
            self._initialize_stats(signals)
            return DetectorStatus.INACTIVE
        if self._count < self.n_training:
            self._update_stats(signals)
            return DetectorStatus.INACTIVE
        status = self._evaluate_signals(signals)
        return status

    def update_status(self, state) -> DetectorStatus:
        """Update the detector status with the given state.

        For the first `n_training` calls, only update the baseline for future
        detection.  Afterwards, attempt to detect events.

        Parameters
        ----------
        state: state-like object
            An object with a `hoomd.Snapshot` like API. Examples include
            `gsd.hoomd.Frame` and `hoomd.Snapshot`. This is used to pass to
            generator to return the corresponding signals.
        """
        status = self._update_status(state)
        self._count += 1
        return status

    def event_details(self) -> List[Tuple[int, List[str]]]:
        """If an event has been detected provide information about the event."""
        return self._events

    @property
    def generators(self) -> Tuple[Generator, ...]:
        """tuple[signal.Generator] Current generators used by the detector."""
        return self._generators

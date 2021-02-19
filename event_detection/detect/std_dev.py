"""Use the standard deviation of signals from a baseline to detect events."""

from abc import ABC, abstractmethod
from copy import copy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .. import event
from ..signal import Generator, Signal, SignalType
from . import Detector, DetectorStatus


class _SignalStats(ABC):
    @abstractmethod
    def update_baseline(self, entry: Signal) -> None:
        pass

    @abstractmethod
    def compute(self):
        pass

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

    @abstractmethod
    def __getitem__(self, index: Union[int, slice]):
        return copy(self._baseline[index])

    @abstractmethod
    def deviations(self, entry):
        pass


class _ScalarSignalStats(_SignalStats):
    def __init__(self) -> None:
        self._baseline = []
        self.active_counter = 0
        self._recompute_stats = True

    def update_baseline(self, entry: float) -> None:
        self._baseline.append(entry.data)
        self._recompute_stats = True

    def compute(self):
        baseline = np.array(self._baseline)
        self._mean = np.mean(baseline)
        self._std_dev = np.std(baseline)

    def __getitem__(self, index: Union[int, slice]):
        return copy(self._baseline[index])

    def deviations(self, entry: Signal) -> float:
        return abs(entry.data - self.mean) / self.std_dev


class _ArraySignalStats(_SignalStats):
    def __init__(self, percentiles: List[int]) -> None:
        self.percentiles = percentiles
        self.percentile_indices = None
        self._recompute_stats = True
        self.active_counter = np.zeros(len(percentiles))
        self._baseline = []

    def update_baseline(self, entry: Signal) -> None:
        if entry.type != SignalType.ARRAY:
            raise ValueError("Signal type and SignalStats type do not align.")
        if self.percentile_indices is None:
            self.percentile_indices = [
                int(percent / 100.0 * (len(entry.data) - 1))
                for percent in self.percentiles
            ]
        self._baseline.append(np.sort(entry.data)[self.percentile_indices])
        self._recompute_stats = True

    def compute(self) -> None:
        baselines = np.array(self._baseline)
        self._mean = np.mean(baselines, axis=0)
        self._std_dev = np.std(baselines, axis=0)

    def deviations(self, entry: Signal) -> np.ndarray:
        return (
            np.abs(np.sort(entry.data)[self.percentile_indices] - self.mean)
            / self.std_dev
        )


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
        signals: List[Generator],
        n_training: int = 50,
        n_std_dev: float = 2.0,
    ):
        """Create a `StdDevDetector` object.

        Parameters
        ----------
        signals: list[Generator]
            The signals to use for event detection.
        n_training: int
            The number of calls to use for generating the baseline before
            attempting to detect a signal.
        n_stdev: float
            The number of standard deviations from the base line that consitutes
            triggering an event.
        """
        self._signals = signals
        self.n_training = n_training
        self.n_std_dev = 2.0
        self._signals_stats = {}
        self._count = 0
        self.percentages = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
        self.status = DetectorStatus.INACTIVE

    def _get_signals(self, state) -> Dict[str, Signal]:
        signals = {}
        for generator in self.signals:
            new_signals = generator.generate(state)
            if signals.keys().isdisjoint(new_signals):
                signals.update(new_signals)
            else:
                raise RuntimeError("All signals must have unique names.")
        return signals

    def _update_stats(self, signals: Dict[str, Signal]) -> None:
        for signal_name, signal in signals.items():
            self._signals_stats[signal_name].update_baseline(signal)

    def _initialize_stats(self, signals: Dict[str, Signal]) -> None:
        for signal_name, signal in signals.items():
            if signal.type == SignalType.SCALAR:
                signal_stats = _ScalarSignalStats()
            elif signal.type == SignalType.ARRAY:
                signal_stats = _ArraySignalStats(self.percentages)

            signal_stats.update_baseline(signal)
            self._signals_stats[signal_name] = signal_stats

    def _evaluate_signals(self, signals):
        max_event_counter = 0
        event_happened = False
        for signal_name, signal in signals.items():
            signal_stats = self._signals_stats[signal_name]
            deviations = signal_stats.deviations(signal)
            if isinstance(deviations, np.ndarray):
                events = deviations > self.n_std_dev
                event_detected = any(events)
                if event_detected:
                    signal_stats.active_counter[events] += 1
                    max_event_counter = max(
                        max_event_counter, signal_stats.active_counter.max()
                    )
                else:
                    signal_stats.active_counter[:] = 0
            else:
                event_detected = deviations > self.n_std_dev
                if event_detected:
                    signal_stats.active_counter += 1
                    max_event_counter = max(
                        max_event_counter, signal_stats.active_counter
                    )
                else:
                    signal_stats.active_counter = 0
            event_happened = event_happened or event_detected
        if max_event_counter > 10:
            self.status = DetectorStatus.CONFIRMED
        elif event_happened:
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

    def event_details(self) -> Optional[event.Event]:
        """If an event has been detected provide information about the event."""
        raise NotImplementedError

    @property
    def signals(self) -> Tuple[Generator, ...]:
        """tuple[signal.Signal] All current signals used for the detector."""
        return self._signals

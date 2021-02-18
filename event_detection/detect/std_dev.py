"""Use the standard deviation of signals from a baseline to detect events."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .. import event
from ..signal import Generator
from . import Detector, DetectorStatus


class StdDevDetector(Detector):
    """Detects events using the standard deviation of a signal from a baseline.

    For array signals percentiles are taken and used as individual distributions
    to measure the baseline for.

    Attributes
    ----------
    n_training: int
        The number of calls to use for generating the baseline before attempting
        to detect a signal.
    n_stop: int
        The number of steps to use for generating the baseline. If this is
        greater than `n_training`, then the instance will attempt to detect
        events for calls `n_traning` to ``n_training + n_stop``. If no event is
        detected then the baseline will continue to be updated until `n_stop`
        calls have been made.
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
        n_stop: int = 100,
        n_stddev: float = 2.0,
    ):
        """Create a `StdDevDetector` object.

        Parameters
        ----------
        signals: list[Generator]
            The signals to use for event detection.
        n_training: int
            The number of calls to use for generating the baseline before
            attempting to detect a signal.
        n_stop: int
            The number of steps to use for generating the baseline. If this is
            greater than `n_training`, then the instance will attempt to detect
            events for calls `n_traning` to ``n_training + n_stop``. If no event
            is detected then the baseline will continue to be updated until
            `n_stop` calls have been made.
        n_stdev: float
            The number of standard deviations from the base line that consitutes
            triggering an event.
        """
        self._signals = signals
        self.n_training = n_training
        self.n_stop = n_stop
        self.n_stddev = 2.0
        self._signals_stats = {}
        self._count = 0
        self.status = DetectorStatus.INACTIVE

    def _get_signals(self, state) -> Dict[str, Union[float, np.ndarray]]:
        signals = {}
        for generator in self.signals:
            new_signals = generator.query(state)
            if signals.keys().isdisjoint(new_signals):
                signals.update(new_signals)
            else:
                raise RuntimeError("All signals must have unique names.")
        return signals

    def _update_stats(
        self, signals: Dict[str, Union[float, np.ndarray]]
    ) -> None:
        for signal_name, signal in signals.items():
            signal_stats = self._signals_stats[signal_name]
            if isinstance(signal, np.ndarray):
                sorted_signal = np.sort(signal)
                indices = signal_stats["_percent_indices_pairs"]
                for percent, index in indices:
                    signal_stats[percent]["values"].append(sorted_signal[index])
            else:
                signal_stats["values"].append(signal)

    def _initialize_stats(
        self, signals: Dict[str, Union[float, np.ndarray]]
    ) -> None:
        percentages = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
        for signal_name, signal in signals.items():
            signal_stats = self._signals_stats[signal_name] = {}
            if isinstance(signal, np.ndarray):
                signal_stats["_percent_indices_pairs"] = []
                for percent in percentages:
                    index = int(percent / 100.0 * (len(signal) - 1))
                    signal_stats["_percent_indices_pairs"].append(
                        (percent, index)
                    )
                    signal_stats[percent] = {"values": [signal[index]]}
            else:
                signal_stats["values"] = [signal]

    def _compute_statistics(self):
        for signal_stats in self._signals_stats.values():
            if "values" in signal_stats:
                values = np.array(signal_stats["values"])
                signal_stats["mean"] = np.mean(values)
                signal_stats["std_dev"] = np.std(values)
            else:
                for name, percent_stats in signal_stats.items():
                    if not isinstance(name, int):
                        continue
                    values = np.array(percent_stats["values"])
                    percent_stats["mean"] = np.mean(values)
                    percent_stats["std_dev"] = np.std(values)

    def _evaluate_signals(self, signals):
        for signal_name, signal in signals.items():
            signal_stats = self._signals_stats[signal_name]
            if isinstance(signal, np.ndarray):
                sorted_signal = np.sort(signal)
                indices = signal_stats["_percent_indices_pairs"]
                for percent, index in indices:
                    if not isinstance(percent, int):
                        continue
                    deviation = abs(
                        sorted_signal[index] - signal_stats[percent]["mean"]
                    )
                    std_dev = signal_stats[percent]["std_dev"]
                    if (deviation - (self.n_stddev * std_dev)) > 0:
                        if self.status == DetectorStatus.INACTIVE:
                            self.status = DetectorStatus.ACTIVE
            else:
                deviation = abs(signal - signal_stats["mean"])
                std_dev = signal_stats["std_dev"]
                if (deviation - (self.n_stddev * std_dev)) > 0:
                    if self.status == DetectorStatus.INACTIVE:
                        self.status = DetectorStatus.ACTIVE
            return self.status

    def _update_status(self, state) -> DetectorStatus:
        signals = self._get_signals(state)
        if self._count == 0:
            self._initialize_stats(signals)
            return DetectorStatus.INACTIVE
        if self._count < self.n_training:
            self._update_stats(signals)
            return DetectorStatus.INACTIVE
        if self._count == self.n_training:
            self._compute_statistics()
        status = self._evaluate_signals(signals)
        if self._count < self.n_stop and status < DetectorStatus.ACTIVE:
            self._update_stats(signals)
            self._compute_statistics()
        return status

    def update_status(self, state) -> DetectorStatus:
        """Update the detector status with the given state.

        For the first `n_training` calls, only update the baseline for future
        detection. From ``n_training + 1`` to ``n_training + n_stop`` calls
        update baseline while also attempting to detect the signal. Afterwards,
        only attempt to detect events.

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

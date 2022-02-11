"""Use the standard deviation of signals from a baseline to detect events."""

from typing import Any, Dict

import dupin.data as data

from . import DetectorStatus


class StdDevDetector:
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
    events: list[tuple[int, list[str]]]
        A list where each entry is a "verified" event with the first entry in
        the tuple representing the n-th call where the event was detected and
        the second is a list of feature names that triggered the detector.
    """

    def __init__(
        self,
        generator: [data.base.GeneratorLike],
        n_training: int = 50,
        n_std_dev: float = 2.0,
        threshold: int = 10,
    ):
        """Create a `StdDevDetector` object.

        Parameters
        ----------
        generators: list[data.base.Generator]
            The generators used for signal generation to use for event
            detection.
        n_training: `int`, optional
            The number of calls to use for generating the baseline before
            attempting to detect a signal.
        n_stdev: `float`, optional
            The number of standard deviations from the base line that consitutes
            triggering an event.
        theshold: `int`, optional
            The number of consequative calls that a signal is over/under the set
            number of standard deviations.
        """
        self._generator = generator
        self._aggregator = data.SignalAggregator(generator)
        self.n_training = n_training
        self.n_std_dev = n_std_dev
        self._count = 0
        self._threshold = threshold
        self.events = []
        self.status = DetectorStatus.INACTIVE

    def _evaluate_signals(self, snapshot: Dict[str, float]):
        max_event_counter = 0
        confirmed_signals = []
        # Check each signal and update confirmed_signals list and all activity
        # counters
        for signal_name, signal in snapshot.items():
            deviation = abs(self._means[signal_name] - signal)
            if deviation / self._std_dev[signal_name] > self.n_std_dev:
                current_counter = self._active_counters[signal_name] + 1
                self._active_counters[signal_name] = current_counter
                if current_counter > max_event_counter:
                    max_event_counter = current_counter
                if current_counter > self._threshold:
                    confirmed_signals.append(signal_name)
            else:
                self._active_counters[signal_name] = 0

        # Check for detector status
        if len(confirmed_signals) > 0:
            self.status = DetectorStatus.CONFIRMED
            self.events.append(self._count, confirmed_signals)
        elif max_event_counter > 0:
            self.status = DetectorStatus.ACTIVE
        else:
            self.status = DetectorStatus.INACTIVE
        return self.status

    def _update_status(self, *args: Any, **kwargs: Any) -> DetectorStatus:
        """Update the detector one step with given arguments."""
        # compute the new index's features
        self._aggregator.accumulate(*args, **kwargs)
        # before training is done just return inactive
        if self._count < self.n_training:
            return DetectorStatus.INACTIVE
        # Compute std_dev and mean when training is finished
        if self._count == self.n_training:
            signals_df = self._aggregator.to_dataframe()
            self._means = signals_df.mean(axis=1).to_dict()
            self._std_devs = signals_df.std(axis=1).to_dict()
            self._active_counters = {name: 0 for name in self._means}
        # Attempt to detect events pasted training
        status = self._evaluate_signals(self._aggregator.signals[-1])
        return status

    def update_status(self, *args: Any, **kwargs: Any) -> DetectorStatus:
        r"""Update the detector status with the given arguments.

        For the first `n_training` calls, only update the baseline for future
        detection.  Afterwards, attempt to detect events.

        Parameters
        ----------
        \*args:
            positional arguments to pass to ``self.generator.__call__``.
        \*\*kwargs:
            keyword arguments to pass to ``self.generator.__call__``.

        Returns
        -------
        status: dupin.detect.online.DetectorStatus
            The status of the detector (whether an event has been detected).
        """
        status = self._update_status(*args, **kwargs)
        self._count += 1
        return status

    @property
    def generator(self) -> data.base.Generator:
        """tuple[data.base.Generator] generators used by the detector."""
        return self._generator

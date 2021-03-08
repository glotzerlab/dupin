"""Provide base API for rare event detectors."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

from event_detection import event, signal
from event_detection.util import OrderedEnum


class DetectorStatus(OrderedEnum):
    """The current status of a detector."""

    INACTIVE = 1
    ACTIVE = 2
    CONFIRMED = 3


class Detector(ABC):
    """Abstract base class for rare event detectors.

    A `Detector` takes in one or more `signal.Generator` instances and uses them
    to attempt to detect an _event_ occurring. All `Detector` subclasses must
    follow a 3 state paradigm: inactive, active, confirmed. The states are
    roughly:

    * inactive - No anamolous signal detected
    * active - Signal detected, but make be temporary fluctuation
    * confirmed - Detected signal has persisted or increased confirming
      detection

    What consitutes a transition from inactive to active state or active to
    confirmed is up to the particular subclass.

    All subclasses must implement three things a `update_status` method that
    updates its internal state and returns a `DetectorStatus` that state, a
    `event_details` method that returns an `event.Event` instance or ``None``
    if no event was detected yet, and a `generators` property that returns a
    sequence of currently used `signal.Generator` objects.
    """

    @abstractmethod
    def update_status(self, state) -> DetectorStatus:
        """Update detector state given the current state information.

        Parameters
        ----------
        state: state-like object
            An object with a `hoomd.Snapshot` like API. Examples include
            `gsd.hoomd.Frame` and `hoomd.Snapshot`. This is used to pass to
            signals to return the individual signal.

        Returns
        status: DetectorStatus
            The current status of the detector.
        """
        pass

    @abstractmethod
    def event_details(self) -> Optional[event.Event]:
        """Return the details of an event if detected.

        Returns
        -------
        event.Event or None
            The detected event or a ``NoneType`` object.
        """
        pass

    @property
    @abstractmethod
    def generators(self) -> Tuple[signal.Generator, ...]:
        """tuple[signal.Generator] All current signals used for the detector."""
        pass

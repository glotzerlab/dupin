"""Provide base API for rare event detectors."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Tuple

from event_detection import data


class OrderedEnum(Enum):
    """An enum that can use the comparison operators."""

    def __ge__(self, other):  # noqa: D105
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

    def __gt__(self, other):  # noqa: D105
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __le__(self, other):  # noqa: D105
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

    def __lt__(self, other):  # noqa: D105
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


class DetectorStatus(OrderedEnum):
    """The current status of a detector."""

    INACTIVE = 1
    ACTIVE = 2
    CONFIRMED = 3


class Detector(ABC):
    """Abstract base class for rare event detectors.

    A `Detector` takes in one or more `data.Generator` instances and uses
    them to attempt to detect an _event_ occurring. All `Detector` subclasses
    must follow a 3 state paradigm: inactive, active, confirmed. The states are
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
    sequence of currently used `data.Generator` objects.
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
    def event_details(self) -> List[Tuple[int, List[str]]]:
        """Return the details of an event if detected.

        Returns
        -------
        event.Event or None
            The detected event or a ``NoneType`` object.
        """
        pass

    @property
    @abstractmethod
    def generators(self) -> Tuple[data.Generator, ...]:
        """tuple[data.Generator] All current signals used for the detector."""
        pass

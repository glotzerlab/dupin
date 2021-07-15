"""Define the interface for signals used for event detection."""

from abc import ABC, abstractmethod
from typing import Dict


class Generator(ABC):
    """The abstract base class for signals used for event detection.

    This just defines a simple `genrate` interface where a `detect.Detector`
    subclass can generate a signal for a given input. These are essentially
    features or descriptors that are placed in a particular API.
    """

    @abstractmethod
    def generate(self, state) -> Dict[str, float]:
        """Return the output signal for a given state.

        Parameters
        ----------
        state: state-like object
            An object with a `hoomd.Snapshot` like API. Examples include
            `gsd.hoomd.Frame` and `hoomd.Snapshot`. This is used to pass to
            generator to return the corresponding signals.

        Returns
        -------
        signals:
            Returns a mapping of signal names to floating point values.
        """
        pass

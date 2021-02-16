"""Define the interface for signals used for event detection."""

from abc import abstractmethod
from abc.collections import Hashable
from typing import Union

import numpy as np


class Signal(Hashable):
    """The abstract base class for signals used for event detection.

    This just defines a simple `query` interface where a `detector.Detector`
    subclass can query information on a given signal for a given input. These
    are essentially features or descriptors that are placed in a particular
    API.  In addition to the `query` methods, `Signal` subclasses should also
    implement `__hash__` and `__eq__` to enable checking that a given signal is
    only used once and enable associating a signal with other information.
    """

    @abstractmethod
    def query(self, state) -> Union(float, np.ndarray):
        """Return the output signal for a given state.

        Parameters
        ----------
        state: state-like object
            An object with a `hoomd.Snapshot` like API. Examples include
            `gsd.hoomd.Frame` and `hoomd.Snapshot`. This is used to pass to
            signals to return the individual signal.

        Returns
        -------
        signal: float or np.ndarray
            Either returns a single value for the entire system or an array of
            values representing localized information about a system state. A
            common use case for arrays would be per-particle descriptors.
        """
        pass

    @abstractmethod
    def __eq__(self, other):
        """Whether or not two signals are equivalent."""
        pass

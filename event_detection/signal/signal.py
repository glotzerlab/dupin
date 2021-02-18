"""Defines wrapper class for denoting the type of signal."""

from collections.abc import Sequence
from typing import Union

import numpy as np

from ..util import OrderedEnum


class SignalType(OrderedEnum):
    """Denote the type of signal.

    * SCALAR: a singular real number.
    * ARRAY: a set number of real numbers.
    * VARIABLE: a variable number of real numbers.
    """

    SCALAR = 1
    ARRAY = 2
    VARIABLE = 3


class Signal:
    """The data associated with a description of a molecular system."""

    def __init__(
        self, data: Union[float, np.ndarray], type_: SignalType = None
    ) -> None:
        """Create a Signal object.

        Parameters
        ----------
        data:
            The raw signal data.
        type_:
            The signal type. If not passed all array-like objects are assumed to
            be of fixed length.
        """
        self.data = data
        self.type = self._detect_type(data) if type_ is None else type_

    def _detect_type(self, data):
        if isinstance(data, (np.ndarray, Sequence)):
            return SignalType.ARRAY
        else:
            return SignalType.SCALAR

"""Provide base API for online detector statuses."""

from enum import Enum


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

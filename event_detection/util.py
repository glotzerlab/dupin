"""General helper functions and classes."""
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

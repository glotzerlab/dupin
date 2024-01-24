"""Methods for event detection in molecular simulations.

dupin provides classes and functions to help in the event detection process but
does not currently have a detection method implemented or independent elbow
detection. The package provides interfaces into `ruptures`_ for an event
detection implementation and `kneed`_ for elbow detection.

This module also provides cost functions for use with `ruptures`_, a scheme for
determining the correct number of events in `SweepDetector` and some elbow
detection helpers.
"""

from .costs import CostLinearBiasedFit, CostLinearFit
from .detect import (
    SweepDetector,
    kneedle_elbow_detection,
    two_pass_elbow_detection,
)

__all__ = (
    "CostLinearFit",
    "CostLinearBiasedFit",
    "SweepDetector",
    "kneedle_elbow_detection",
    "two_pass_elbow_detection",
)

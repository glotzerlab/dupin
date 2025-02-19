# Copyright (c) 2023-2025 The Regents of the University of Michigan.
# This file is from the dupin project, released under the BSD 3-Clause License.

"""Perform various kinds of preprocessing on generated signals.

This module provides resources for the transformation step of the event
detection pipeline.
"""

from . import filter, signal, supervised  # noqa: A004

__all__ = ("filter", "signal", "supervised")

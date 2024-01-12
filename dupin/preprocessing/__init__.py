"""Perform various kinds of preprocessing on generated signals.

This module provides resources for the transformation step of the event
detection pipeline.
"""
from . import filter, signal, supervised

__all__ = ("filter", "signal", "supervised")

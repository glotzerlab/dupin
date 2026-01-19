# Copyright (c) 2023-2026 The Regents of the University of Michigan.
# This file is from the dupin project, released under the BSD 3-Clause License.

"""Top-level package for event-detection."""

__author__ = """Brandon Butler"""
__email__ = "butlerbr@umich.edu"
__version__ = "0.0.1"

from . import data, detect, errors, postprocessing, preprocessing

__all__ = ("data", "detect", "errors", "postprocessing", "preprocessing")

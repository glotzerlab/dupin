"""Top-level package for event-detection."""

__author__ = """Brandon Butler"""
__email__ = "butlerbr@umich.edu"
__version__ = "0.1.0"

from . import data, detect, errors, postprocessing, preprocessing

__all__ = ("data", "detect", "errors", "postprocessing", "preprocessing")

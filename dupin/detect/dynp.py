"""Implements dynamic programming class for optimal segementation algorithm."""
import _DynP
import numpy as np


class DynP:
    """Detects the change points in a time series.

    Attributes
    ----------
    data: np.ndarray
        Matrix storing the time series data.
    num_bkps: int
        Number of change points to detect.
    jump: int
        Interval for checking potential change points. Changing will
        not provide optimal detection, but will reduce runtime.
    min_size: int
        Minimum size of a segment. Changing will not provide optimal
        detection, but will reduce runtime.
    """

    def __init__(
        self, data: np.ndarray, num_bkps: int, jump: int, min_size: int
    ):
        """Initialize the DynamicProgramming instance with given parameters."""
        self.dynp = _DynP.DynamicProgramming(data, num_bkps, jump, min_size)

    def set_num_threads(self, num_threads: int):
        """Set the number of threads for parallelization.

        Parameters
        ----------
        num_threads: int
            The number of threads to use during computation. Default
            is determined automatically.
        """
        self.dynp.set_threads(num_threads)

    def fit(self, num_bkps: int) -> list:
        """Calculate the cost matrix and return the breakpoints.

        Parameters
        ----------
        num_bkps: int
            number of change points to detect.

        Returns
        -------
            list: A list of integers representing the breakpoints.
        """
        return self.dynp.fit()

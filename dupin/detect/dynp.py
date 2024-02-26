"""Implements dynamic programming class for optimal segementation algorithm."""
import _dupin
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


    Methods
    -------
    __init__(self, data: np.ndarray, num_bkps: int, jump: int, min_size: int)
        Initializes the DynamicProgramming instance with the time series data
        and parameters.
    set_num_threads(self, num_threads: int)
        Sets the number of threads to be used for parallel computation.
    fit(self, num_bkps: int) -> list
        Calculates the cost matrix and identifies the optimal breakpoints in
        the time series data.

    Example Usage
    -------------
    >>> import numpy as np
    >>> from dynp import DynP
    >>> data = np.random.rand(100, 1)  # Simulated time series data
    >>> num_bkps = 3  # Number of breakpoints to detect
    >>> jump = 1  # Interval for checking potential breakpoints
    >>> min_size = 3  # Minimum size of a segment
    >>> model = Dynp(data, num_bkps, jump, min_size)
    >>> breakpoints = model.fit(num_bkps)
    >>> print(breakpoints)
    """

    def __init__(
        self, data: np.ndarray, num_bkps: int, jump: int, min_size: int
    ):
        """Initialize the DynamicProgramming instance with given parameters."""
        self._dupin = _dupin.DynamicProgramming(data, num_bkps, jump, min_size)

    def set_num_threads(self, num_threads: int):
        """Set the number of threads for parallelization.

        Parameters
        ----------
        num_threads: int
            The number of threads to use during computation. Default
            is determined automatically.
        """
        self._dupin.set_threads(num_threads)

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
        return self._dupin.fit(num_bkps)

import _DynP

class DynP:
    """Dynamic Programming class for calculating optimal segmentation

    Attributes:
        data (np.ndarray): Matrix storing the dataset.
        num_bkps (int): Number of breakpoints to detect.
        jump (int): Interval for checking potential breakpoints.
        min_size (int): Minimum size of a segment.
    """

    def __init__(self, data: np.ndarray, num_bkps: int, jump: int, min_size: int):
        """Initializes the DynamicProgramming instance with given parameters."""
        self.dynp = _DynP.DynamicProgramming(data, num_bkps, jump, min_size)

    def set_num_threads(self, num_threads: int):
        """Sets the number of threads for parallelization.

        Args:
            num_threads (int): The number of threads to use.
        """
        self.dynp.set_threads(num_threads)

    def return_breakpoints(self) -> list:
        """Returns the optimal set of breakpoints after segmentation.

        Returns:
            list: A list of integers representing the breakpoints.
        """
        return self.dynp.return_breakpoints()

    def initialize_cost_matrix(self):
        """Initializes and fills the upper triangular cost matrix for all data segments."""
        self.dynp.initialize_cost_matrix()

    def fit(self) -> list:
        """Calculates the cost matrix and returns the breakpoints.

        Returns:
            list: A list of integers representing the breakpoints.
        """
        return self.dynp.fit()
    
    
    

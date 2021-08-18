"""Functions and classes for allowing logging extra data from pipelines."""


class Logger:
    """Class for logging extra information from data pipeline."""

    def __init__(self):
        """Construct a Logger instance."""
        self._data = []
        self._current_frame = {}
        self._current_context = None

    def set_context(self, key):
        """Set the current distribution to store information on."""
        self._current_frame.setdefault(key, {})
        self._current_context = self._current_frame[key]

    def __setitem__(self, key, value):
        """Internally store information from data pipeline."""
        self._current_context[key] = value

    def end_frame(self):
        """End the current frame of data. Allows separate by time of data."""
        self._data.append(self._current_frame)
        self._current_frame = {}

    @property
    def frame(self):
        """`list` [`dict`]: Assess a particular frame of data."""
        return self._data

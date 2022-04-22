"""Functions and classes for allowing logging extra data from pipelines."""

import numpy as np

from dupin import errors

try:
    import pandas as pd
except ImportError:
    pd = errors._RaiseModuleError("pandas")


class Logger:
    """Class for logging extra information from data pipeline."""

    def __init__(self):
        """Construct a Logger instance."""
        self._data = []
        self._current_frame = {}
        self._current_context = None
        self._current_key = None

    def set_context(self, key):
        """Set the current distribution to store information on."""
        # is not none or empty
        if self._current_context:
            self._current_frame[self._context_key] = self._current_context
        self._current_context = {}
        self._context_key = key

    def __setitem__(self, key, value):
        """Internally store information from data pipeline."""
        self._current_context[key] = value

    def end_frame(self):
        """End the current frame of data. Allows separate by time of data."""
        # is not none or empty
        if self._current_context:
            self._current_frame[self._context_key] = self._current_context
        self._data.append(self._current_frame)
        self._current_frame = {}

    @property
    def frames(self):
        """`list` [`dict`]: Assess a particular frame of data."""
        return self._data

    def to_dataframe(self):
        """Return a `pandas.DataFrame` object consisting of stored data.

        Warning:
            This assumes the pipeline produces homogenous data along a
            trajectory.
        """
        frame_data = self._data[0]
        column_index = pd.MultiIndex.from_tuples(
            _create_column_index(frame_data)
        )
        data_arr = _log_data_to_array(
            self._data,
            np.empty((len(self._data), len(column_index)), dtype=float),
        )
        return pd.DataFrame(data_arr, columns=column_index)


def _create_column_index(log_data):
    for key, value in log_data.items():
        if isinstance(value, dict):
            for inner_index in _create_column_index(value):
                yield (key,) + inner_index
        else:
            yield (key,)


def _log_data_to_array(data, out):
    def write_frame(frame_data, out, index):
        for value in frame_data.values():
            if isinstance(value, dict):
                index = write_frame(value, out, index)
            else:
                out[index] = value
                index += 1
        return index

    for i, frame in enumerate(data):
        write_frame(frame, out[i], 0)
    return out

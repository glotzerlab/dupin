"""Functions and classes for allowing logging extra data from pipelines.

Logging serves as the only directly supported means of introspecting into the
state of a pipeline. This allows pipeline components to report to a logger
information that otherwise would be discarded as it is not part of the feature
generation. An example would be the index for the specified least and greatest
values in `dupin.data.reduce.NthGreatest`. This particular example allows a user
after feature generation to see which particles in a trajectory are chosen each
step for a feature.

Similar to the pipeline proper, the logging infrastructure expects pipeline
components to return dictionaries. The data itself is stored in a list where
each entry is data from one frame of the trajectory parsed. The list elements
are nested dictionaries where the top level is feature names (at the various
stages of modification) and the second level is the class/modifier specific
identifier, and the third and final level is the logging data for that object
for that frame.

Note
----
    Logging data is not accessible to other parts of the pipeline.
"""

import numpy as np

from .. import errors

try:
    import pandas as pd
except ImportError:
    pd = errors._RaiseModuleError("pandas")


class Logger:
    """Class for logging extra information from data pipeline.

    Stores available metadata from pipeline components. Not all components offer
    metadata, and those that do document them.
    """

    def __init__(self):
        """Construct a Logger instance."""
        self._data = []
        self._reset()

    def _set_context(self, key):
        """Set the current distribution to store information on.

        This sets the name for the current feature being logged.
        """
        # is not none or empty
        if self._current_context:
            self._current_frame.setdefault(
                self._context_key, self._current_context
            )
        # Don't duplicate the same key.
        self._current_context = self._current_frame.get(key, {})
        self._context_key = key

    def __setitem__(self, key, value):
        """Internally store information from data pipeline.

        This is used to store pipeline component feature specific metadata.
        """
        self._current_context[key] = value

    def end_frame(self):
        """End the current frame of data. Allows separate by time of data."""
        # is not none or empty
        if self._current_context:
            self._current_frame.setdefault(
                self._context_key, self._current_context
            )
        self._data.append(self._current_frame)
        self._reset()

    def _reset(self):
        self._current_frame = {}
        self._current_context = None
        self._current_key = None

    @property
    def frames(self):
        r"""`list` [`dict`]: Assess a particular frame of data.

        The data is a `list` of `dict` where keys are features and values are
        `dict`\ s with the metadata gathered from the pipeline components.
        """
        return self._data

    def to_dataframe(self):
        """Return a `pandas.DataFrame` object consisting of stored data.

        This uses `pandas.MultiIndex` to map the nested dictionaries to a
        dataframe.

        Warning
        -------
            This assumes the pipeline produces homogenous data along a
            trajectory.

        Warning
        -------
            This only works for floating point logged values.
        """
        frame_data = self._first_non_empty(self._data)
        if frame_data is None:
            return pd.DataFrame()
        column_index = pd.MultiIndex.from_tuples(
            _create_column_index(frame_data)
        )
        # TODO: Extend to other dtypes?
        data_arr = _log_data_to_array(
            self._data,
            np.empty((len(self._data), len(column_index)), dtype=float),
        )
        return pd.DataFrame(data_arr, columns=column_index)

    @staticmethod
    def _first_non_empty(data: list[dict]):
        for d in data:
            if len(d) > 0:
                return d
        return None


def _create_column_index(log_data):
    """Yield tuples of keys for creating a multi-index from a nested dict."""
    for key, value in log_data.items():
        if isinstance(value, dict):
            for inner_index in _create_column_index(value):
                yield (key, *inner_index)
        else:
            yield (key,)


def _log_data_to_array(data, out):
    """Take the output of `Logger.frames` and fills a NumPy array with the data.

    This is designed to be used with _create_column_index to help the `Logger`
    create a `pandas.Dataframe`.
    """

    def write_frame(frame_data, out, index):
        """Recursive function which fills a row of an array.

        Parameters
        ----------
        frame_data: dict | float
            The data to store.
        out: numpy.ndarray
            The array row to fill.
        index: int
            An index to write the data out to. This is used with recursion to
            ensure that each entry is written once and in a non-overlapping
            location.
        """
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

"""General functions for analyzing change points once detected.

This module is designed to primarly work with logger data and the original
trajectory.
"""

import itertools
import warnings

import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats

import dupin.preprocessing


def _ipairs(sequence, n=2):
    yield from zip(sequence, *[sequence[i:] for i in range(1, n)])


class EventFeatures:
    """Finds features out of a signal that constitute changes.

    The class provides multiple methods for detecting features that explain a
    particular partitioning of space. The class requires change points to work.
    In other words, this is a post detection method for interpretting change
    point detection results.

    Parameters
    ----------
    signal: :math:`(N_{samples}, N_{features})` np.ndarray of float
        The signal to find features from.
    change_points: list[int]
        The change points associated with the signal.
    """

    def __init__(self, signal, change_points):
        self._signal = signal
        self._change_points = [0, *change_points, len(self._signal)]

    def linear(self, sample_size, sensitivity, extend_small_regions=True):
        """Find features that change linearly between a pair of change points.

        Warning:
            This function is designed to events that involve changes within
            pairs of change points. This will not detect mean shift events.

        Parameters
        ----------
        signal: :math:`(N_{samples}, N_{features})` numpy.ndarray of float
            The signal the change points are from.
        change_points: list[int]
            A list of all interior change points (beginning and end of signal
            not included).
        sample_size: float or int, optional
            Either the fraction of the overall signal to use to evaluate the
            statistics of each end of all subsignals defined by the change
            points, or the number of data points to use on each end of the
            signal for statistics. Default to 0.1. If this would result in less
            than three data points, three will be used.
        sensitivity: float, optional
            The minimum likelihood that one of the signal's end's mean is drawn
            from the Gaussian approximation of the other end to require. In
            other words, the lower the number the increased probability that the
            difference in means is not random.
        extend_small_regions: bool, optional
            Whether to augment small regions (length less than 6) with previous
            and following segment (if available) to allow for analysis of them.
            Defaults to ``True``.

        Returns
        -------
        features: list [`numpy.ndarray` of float]
            Returns a list of Boolean arrays that filter the original data into
            participating features during each interval. A value of ``None`` is
            used for all intervals that are too small to analyze.
        """
        min_change_point_distance = 7

        def extend_region(beg, end):
            needed_extension = 7 - (end - beg)
            if needed_extension <= 0:
                return beg, end

            if beg > 0:
                beg = max(0, beg - needed_extension // 2)
            new_extension = 7 - (end - beg)
            if end < len(self._signal):
                end = min(len(self._signal), end + new_extension)
            if end - beg < min_change_point_distance:
                warnings.warn(
                    "Subsignal could not be extended enough for analysis.",
                    stacklevel=2,
                )
            return beg, end

        mean_shift = dupin.preprocessing.filter.MeanShift(sensitivity)

        features = []
        for start, end in _ipairs(self._change_points):
            if extend_small_regions:
                i, j = extend_region(start, end)
            else:
                i, j = start, end
            if end - start < min_change_point_distance:
                features.append(None)
                continue

            features.append(
                mean_shift(self._signal[i:j], sample_size, return_filter=True)
            )
        return features

    def mean_shift(self, sensitivity=0.01):
        """Find features that have a shift in mean between change points.

        Parameters
        ----------
        sensitivity: float, optional
            The minimum likelihood that one of the mean of one section is drawn
            from the Gaussian approximation of the other section. In
            other words, the lower the number the increased probability that the
            difference in means is not random. Defaults to 0.01.

        Returns
        -------
        features: list [`numpy.ndarray` of float]
            Returns a list of Boolean arrays that filter the original data into
            participating features during each interval. A value of ``None`` is
            used for all change_points that are too small to analyze.
        """
        features = []

        min_change_point_distance = 3
        for beg, mid, end in _ipairs(self._change_points, 3):
            if min(mid - beg, end - mid) < min_change_point_distance:
                features.append(None)
                continue
            likelihoods = dupin.preprocessing.filter.MeanShift._get_likelihood(
                dupin.preprocessing.filter.MeanShift._get_mean_shift_std(
                    self._signal[beg:mid, :], self._signal[mid:end, :]
                )
            )
            features.append(likelihoods < sensitivity)
        return features

    def inter_linear(self, threshold=0.1):
        """Find features that have a shift in linear fit between change points.

        The method searches for any features whose fitted slope and intercept
        change above ``threshold`` between change points.

        Parameters
        ----------
        threshold: float, optional
            The percentile change between change points to indicate that a
            feature changes between change points. Defaults to 0.1.

        Returns
        -------
        features: list [`numpy.ndarray` of float]
            Returns a list of Boolean arrays that filter the original data into
            participating features during each interval. A value of ``None`` is
            used for all change_points that are too small to analyze.
        """

        def get_line_features(start, end):
            m = np.empty(self._signal.shape[1])
            b = np.empty(self._signal.shape[1])
            for i in range(self._signal.shape[1]):
                line = sp.stats.linregress(
                    np.arange(start, end), self._signal[start:end, i]
                )
                m[i] = line.slope
                b[i] = line.intercept
            return m, b

        def max_percent_diff(a, b):
            abs_diff = np.abs(b - a)
            return abs_diff / np.maximum(a, b)

        features = []
        for beg, mid, end in _ipairs(self._change_points, 3):
            m1, b1 = get_line_features(beg, mid)
            m2, b2 = get_line_features(mid, end)
            percentile_m_shift = max_percent_diff(m1, m2)
            percentile_b_shift = max_percent_diff(b1, b2)
            features.append(
                (percentile_m_shift > threshold)
                | (percentile_b_shift > threshold)
            )
        return features


def retrieve_positions(log_df, trajectory):
    """Retreive positions from data pipeline logger data frame.

    Note:
        This is for use with the `dupin.data.reduce.NthGreatest` and
        `dupin.data.reduce.Percentile` and `dupin.data.logging.Logger`.

    This function can find the positions of particles that reducers picked using
    a logger's dataframe and the positions.

    Parameters
    ----------
    log_df : pandas.Dataframe
        A dataframe taken from a `dupin.data.logging.Logger`.
    trajectory : :math:`(N_{frames}, N_p, 3)` numpy.ndarray of float
        The positions of the particles along the trajectory.

    Returns
    -------
    pandas.Dataframe
        The positions of the particles given the indices of ``log_df``. The
        dataframe uses an multi-index to keep the same columns as ``log_df``
        with the addition of a index level with x,y, and z columns.
    """
    column_mask = np.array(
        [
            any(name in {"Percentile", "NthGreatest"} for name in column)
            for column in log_df.columns
        ],
        dtype=bool,
    )
    num_indices = column_mask.sum()
    positions = np.empty((len(log_df), num_indices * 3), dtype=float)
    for i, snapshot_positions in enumerate(trajectory):
        positions[i, :] = snapshot_positions[
            log_df.iloc[i, column_mask].to_numpy().astype(int)
        ].ravel()
    new_columns = pd.MultiIndex.from_tuples(
        (*original_column, coord)
        for original_column, coord in itertools.product(
            log_df.columns, ("x", "y", "z")
        )
    )
    return pd.DataFrame(positions, columns=new_columns)

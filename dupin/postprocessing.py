"""General functions for analyzing change points once detected."""

import itertools
import warnings

import numpy as np
import pandas as pd

import dupin.preprocessing


def compute_features_in_event(
    signal, change_points, sample_size, sensitivity, extend_small_regions=True
):
    """Compute the participating features within a pair of change points.

    Warning:
        This function is designed to work with a linear cost function, and
        assumes change points denote differences in linear signals (e.g. not
        mean-shift signals).

    Internally this function uses
    `dupin.preprocessing.filter.mean_shift`.

    Parameters
    ----------
    signal: :math:`(N_{samples}, N_{features})` numpy.ndarray of float
        The signal the change points are from.
    change_points: list[int]
        A list of all interior change points (beginning and end of signal not
        included).
    sample_size: float or int, optional
        Either the fraction of the overall signal to use to evaluate the
        statistics of each end of all subsignals defined by the change points,
        or the number of data points to use on each end of the signal for
        statistics. Default to 0.1. If this would result in less than three data
        points, three will be used.
    sensitivity: float, optional
        The minimum likelihood that one of the signal's end's mean is drawn from
        the Gaussian approximation of the other end to require. In other words,
        the lower the number the increased probability that the difference in
        means is not random. Defaults to 0.01.
    extend_small_regions: bool, optional
        Whether to augment small regions (length less than 6) with previous and
        following segment (if available) to allow for analysis of them.
        Defaults to ``True``.

    Returns
    -------
    participating_features: list [`numpy.ndarray` of float]
        Returns a list of Boolean arrays that filter the original data into
        participating features during each interval. A value of ``None`` is used
        for all intervals that are too small to analyze.
    """
    augmented_change_points = [0] + change_points + [len(signal)]
    participating_features = []

    def extend_region(beg, end):
        needed_extension = 7 - (end - beg)
        if needed_extension <= 0:
            return beg, end

        if beg > 0:
            beg = max(0, beg - needed_extension // 2)
        new_extension = 7 - (end - beg)
        if end < len(signal):
            end = min(len(signal), end + new_extension)
        if end - beg < 7:
            warnings.warn(
                "Subsignal could not be extended enough for analysis."
            )
        return beg, end

    for beg, end in zip(augmented_change_points, augmented_change_points[1:]):
        if extend_small_regions:
            print(beg, end)
            beg, end = extend_region(beg, end)
            print(beg, end)
        try:
            section_features = dupin.preprocessing.filter.mean_shift(
                signal[beg:end], sample_size, sensitivity, return_filter=True
            )
        # If signal is too small, then we just append None as an indicator.
        except ValueError:
            participating_features.append(None)
        else:
            participating_features.append(section_features)
    return participating_features


def retrieve_positions(log_df, trajectory):
    """Retreive positions from data pipeline logger data frame."""
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
        original_column + (coord,)
        for original_column, coord in itertools.product(
            log_df.columns, ("x", "y", "z")
        )
    )
    return pd.DataFrame(positions, columns=new_columns)

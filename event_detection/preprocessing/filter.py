"""Filters to reduce the dimensions of the signal."""

from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import scipy


def _get_sample_size(signal_len: int, sample_size: Optional[Union[int, float]]):
    # 3 on each side is the minimum to perform this analysis.
    if signal_len < 7:
        raise ValueError("Signal to small to perform statistical analysis on.")
    if isinstance(sample_size, float):
        sample_size = max(3, int(sample_size * signal_len))

    if sample_size >= signal_len / 2:
        raise ValueError(
            f"Cannot use {sample_size} frames with a signal of "
            f"length {signal_len}."
        )
    return sample_size


def mean_shift(
    signal: npt.ArrayLike,
    sample_size: Union[int, float] = 0.1,
    sensitivity=0.01,
    return_filter: Optional[bool] = False,
):
    """Filter out dimensions that don't undergo a significant shift in mean.

    The filter computes the mean and standard deviation of both ends of the
    signal, and determines whether the mean of one either end is statistically
    significant (judged by ``sensitivity``) compared to the other. The filter
    assumes Gaussian noise.

    Parameters
    ----------
    signal : :math:`(N_{samples}, N_{features})` numpy.ndarray of float
        The signal to filter dimensions from.
    sample_size: float or int, optional
        Either the fraction of the overall signal to use to evaluate the
        statistics of each end of the signal, or the number of data points to
        use on each end of the signal for statistics. Default to 0.1. If this
        would result in less than three data points, three will be used.
    sensitivity: float, optional
        The minimum likelihood that one of the signal's end's mean is drawn from
        the Gaussian approximation of the other end to require. In other words,
        the lower the number the increased probability that the difference in
        means is not random. Defaults to 0.01.
    return_filter: bool, optional
        Whether to return the Boolean array filter rather than the filtered
        data. Defaults to ``False``.

    Returns
    -------
    :math:`(N_{samples}, N_{filtered})` numpy.ndarray of float or \
            :math:`(N_{features})` numpy.ndarray of bool:
        By default returns the filtered data with features deemed insignificant
        removed. If ``return_filter`` is ``True``, the Boolean array filtering
        features is returned.
    """
    n_frames = _get_sample_size(len(signal), sample_size)
    left_half, right_half = signal[:n_frames], signal[-n_frames:]

    left_mean = np.mean(left_half, axis=0)
    left_std = np.std(left_half, axis=0)
    right_mean = np.mean(right_half, axis=0)
    right_std = np.std(right_half, axis=0)

    sqrt_2 = np.sqrt(2)

    n_std_left = np.abs(right_mean - left_mean) / left_std
    left_likelihood = 1 - scipy.special.erf(n_std_left / sqrt_2)
    n_std_right = np.abs(left_mean - right_mean) / right_std
    right_likelihood = 1 - scipy.special.erf(n_std_right / sqrt_2)

    filter_ = np.logical_or(
        left_likelihood <= sensitivity, right_likelihood <= sensitivity
    )
    if return_filter:
        return filter_
    return signal[filter_]

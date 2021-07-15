"""Functions for smoothing a signal."""


import numpy as np
import scipy as sp


def moving_average(y: np.ndarray, span: int = 1) -> np.ndarray:
    """Smooth an array via a moving average.

    For multidimensional arrays, the smoothing is done on the first axis. This
    is consistent when rows represent multiple variables or features and columns
    represent different instances.
    Parameters
    ----------
    y : input array to smooth.
    span : int, optional

    Returns
    -------
    smoothed_array: np.ndarray
    """
    cumulative_sum = np.cumsum(y, axis=0)
    average = np.empty(y.shape, dtype=y.dtype)
    average[span:-span] = (
        cumulative_sum[2 * span :] - cumulative_sum[: -2 * span]
    ) / (2 * span)
    for i in range(span):
        average[i] = np.average(y[: span + i], axis=0)
        average[-(i + 1)] = np.average(y[-(span + i) :], axis=0)
    return average


def fft_smoothing(y: np.ndarray, cut_off_percentage: int = 0.05):
    """Smooth out the least contributing frequencies of a signal."""
    w = sp.fft.rfft(y, axis=0)
    spectrum = w ** 2
    w[spectrum < (cut_off_percentage * spectrum.max(axis=0))] = 0
    return sp.fft.irfft(w, len(y), axis=0)


def savgol_filter(*args, **kwargs):
    """Alias for `scipy.signal.savgol_filter`."""
    return sp.signal.savgol_filter(*args, **kwargs)

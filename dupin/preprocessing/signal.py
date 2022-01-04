"""Functions for smoothing a signal.

`scipy.signal` provides many additional methods for signal processing that can
be used to transform data in numerous ways.
"""

import bottleneck as bn
import numpy as np
import scipy as sp


def moving_average(y: np.ndarray, span: int = 1) -> np.ndarray:
    """Smooth an array via a moving average.

    For multidimensional arrays, the smoothing is done on the first axis. This
    is consistent when columns represent multiple variables or features and
    rows represent different instances.

    Parameters
    ----------
    y : np.ndarray
        input array to smooth.
    span : int, optional
        The window size to compute the moving mean over. Defaults to 1 which
        does nothing.

    Returns
    -------
    smoothed_array: np.ndarray
    """
    if span % 2 == 0:
        raise ValueError("The window span must be an odd number.")
    if span == 1:
        return np.copy(y)

    half_span = span // 2
    average = np.empty(y.shape, dtype=y.dtype)
    average[half_span:-half_span] = bn.move_mean(y, span)[2 * half_span :]
    for i in range(half_span):
        average[i] = np.mean(y[: half_span + i + 1], axis=0)
        average[-(i + 1)] = np.mean(y[-(half_span + i + 1) :], axis=0)
    return average


def fft_smoothing(
    y: np.ndarray, cut_off_percentage: float = 0.05
) -> np.ndarray:
    """Smooth out the least contributing frequencies of a signal.

    Parameters
    ----------
    y: :math:`(N, M)` numpy.ndarray of float
        The signal to remove low contributing frequencies from. The FFT is
        performed on the first dimension.
    cut_off_percentage: float, optional
        The fraction of the maximum signal in the FFT below which the signal
        should be zeroed out.
    """
    w = sp.fft.rfft(y, axis=0)
    spectrum = w ** 2
    w[spectrum < (cut_off_percentage * spectrum.max(axis=0))] = 0
    return sp.fft.irfft(w, len(y), axis=0)

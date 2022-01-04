"""Functions for smoothing a signal.

`scipy.signal` provides many additional methods for signal processing that can
be used to transform data in numerous ways.
"""

from typing import Any

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
    y: np.ndarray,
    max_frequency: float,
    sampling_frequency: float,
    **kwargs: Any
) -> np.ndarray:
    r"""Smooth out the highest frequencies of a signal.

    Parameters
    ----------
    y: :math:`(N, M)` numpy.ndarray of float
        The signal to remove low contributing frequencies from. The FFT is
        performed on the first dimension.
    max_frequency: float
        The maximum frequency in the signal to allow. The unit for frequency
        must be consistent with ``sampling_frequency``.
    sampling_frequency: float
        The sampling frequency. The unit for frequency must be consistent with
        ``max_frequency``.
    \*\*kwargs:
        Key word arguments to pass to `scipy.signal.ellip`.
    """
    ellip_kwargs = {
        "N": 8,
        "rp": 1,
        "rs": 100,
        "Wn": max_frequency,
        "fs": sampling_frequency,
        "btype": "lowpass",
        "output": "sos",
    }
    ellip_kwargs.update(kwargs)
    sos = sp.signal.ellip(**ellip_kwargs)
    return sp.signal.sosfilt(sos, y)

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
    y : numpy.ndarray
        input array to smooth.
    span : `int`, optional
        The window size to compute the moving mean over. Defaults to 1 which
        does nothing.

    Returns
    -------
    smoothed_array: numpy.ndarray
    """
    if span % 2 == 0:
        msg = "The window span must be an odd number."
        raise ValueError(msg)
    if span == 1:
        return np.copy(y)

    half_span = span // 2
    average = np.empty(y.shape, dtype=y.dtype)
    average[half_span:-half_span] = bn.move_mean(y, span)[2 * half_span :]
    for i in range(half_span):
        average[i] = np.mean(y[: half_span + i + 1], axis=0)
        average[-(i + 1)] = np.mean(y[-(half_span + i + 1) :], axis=0)
    return average


def high_frequency_smoothing(
    y: np.ndarray,
    max_frequency: float,
    **kwargs: Any,
) -> np.ndarray:
    r"""Smooth out the highest frequencies of a signal.

    The code by defaults performs an 8th order Butterworth filter with a
    passband roughly from 0 to ``max_frequency``. We treat the signal as having
    a sampling frequency of 1Hz, so the max_frequency cannot be greater than
    1.0.

    Parameters
    ----------
    y: :math:`(N, M)` numpy.ndarray of float
        The signal to remove low contributing frequencies from. The FFT is
        performed on the first dimension.
    max_frequency: float
        The first frequency to where the gain dips below -0.01 (unless another
        number is specified for ``rp`` in the key word arguments).
    \*\*kwargs:
        Key word arguments to pass to `scipy.signal.ellip`. This may invalidate
        some documentation assumptions above.
    """
    ellip_kwargs = {
        "N": 8,
        "rp": 0.01,
        "rs": 0.01,
        "Wn": max_frequency,
        "fs": 1.0,
        "btype": "lowpass",
        "output": "sos",
    }
    ellip_kwargs.update(kwargs)
    sos = sp.signal.ellip(**ellip_kwargs)
    return sp.signal.sosfilt(sos, y)

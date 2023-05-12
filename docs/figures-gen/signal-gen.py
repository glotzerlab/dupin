import itertools

import numpy as np


def generate_signal(
    slopes: list[float],
    lengths: list[float],
    intercept: float = 0.0,
    noise: float = 0.0,
) -> np.ndarray:
    signal = np.empty(sum(lengths))
    signal[0 : lengths[0]] = np.linspace(
        intercept, intercept + slopes[0] * lengths[0], lengths[0]
    )
    for m, l, i in zip(slopes[1:], lengths[1:], np.cumsum(lengths)):
        y_start = signal[i - 1]
        signal[i : i + l] = np.linspace(y_start, y_start + m * l, l)
    if noise != 0:
        signal += np.random.default_rng(42).normal(0.0, noise, size=len(signal))
    return signal


slopes = [(0, 0.25, 0.0), (1, 0, 0), (-1, 0, 1)]
intercepts = [0.0, -40, 25]
lengths = itertools.repeat((20, 55, 25))

signal = np.stack(
    [
        generate_signal(m, l, i, 2.5)
        for m, l, i in zip(slopes, lengths, intercepts)
    ],
    axis=1,
)
np.save("signal.npy", signal)

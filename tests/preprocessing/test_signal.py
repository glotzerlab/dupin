import numpy as np
import pytest

from dupin.preprocessing.signal import high_frequency_smoothing, moving_average


@pytest.fixture(params=range(5))
def random_array(rng, request):
    return 2e6 * (rng.random(30) - 0.5)


def test_moving_average_base_properties(random_array):
    smoothed_array = moving_average(random_array)
    assert smoothed_array.shape == random_array.shape
    assert np.allclose(smoothed_array, random_array)
    smoothed_array = moving_average(random_array, 3)
    assert smoothed_array.shape == random_array.shape
    assert smoothed_array.std() < random_array.std()


def test_moving_average_exact():
    array = np.arange(1, 21)
    average = moving_average(array, 3)
    array[0] = array[:2].mean()
    array[-1] = array[-2:].mean()
    assert np.allclose(average, array)


def complex_sq(arr):
    results = arr.conjugate()
    np.multiply(arr, results, out=results)
    return np.real(results)


def test_high_frequency_smoothing(random_array):
    # Pretends random array is over 1 second and that we are filtering out
    # frequencies greater than 10 Hz.
    min_period = 10
    max_frequency = 1 / min_period
    smoothed_array = high_frequency_smoothing(random_array, max_frequency)
    assert smoothed_array.shape == random_array.shape
    # This should be true because the original array is random
    assert smoothed_array.std() < random_array.std()
    assert np.all(
        (complex_sq(np.fft.rfft(smoothed_array))[min_period:])
        < complex_sq(np.fft.rfft(random_array)[min_period:])
    )


def test_high_frequency_smoothing_sin():
    t = np.linspace(0, 1, 1000)
    freq_co = 2 * np.pi * t
    array = np.sin(freq_co * 10) + np.sin(freq_co * 100)
    min_period = 15
    max_frequency = 1 / min_period
    smoothed_array = high_frequency_smoothing(array, max_frequency)
    signal_mag = complex_sq(np.fft.rfft(smoothed_array))
    assert np.argmax(signal_mag) < min_period
    assert (
        complex_sq(np.fft.rfft(smoothed_array)[min_period:]).mean()
        < complex_sq(np.fft.rfft(array)[min_period:]).mean()
    )

import numpy as np
import pytest

from dupin.preprocessing.signal import fft_smoothing, moving_average


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
    return results


def test_fft_smoothing(random_array):
    # Pretends random array is over 1 second and that we are filtering out
    # frequencies greater than 10 Hz.
    smoothed_array = fft_smoothing(random_array, 10, 30)
    assert smoothed_array.shape == random_array.shape
    # This should be true because the original array is random
    assert smoothed_array.std() < random_array.std()
    assert np.all(
        (complex_sq(np.fft.rfft(smoothed_array))[11:])
        < complex_sq(np.fft.rfft(random_array**2)[11:])
    )


def test_fft_smoothing_sin():
    t = np.linspace(0, 1, 1000)
    freq_co = 2 * np.pi * t
    array = np.sin(freq_co * 10) + np.sin(freq_co * 100)
    smoothed_array = fft_smoothing(array, 10, 1000)
    signal_mag = complex_sq(np.fft.rfft(smoothed_array)[11:])
    assert np.argmax(signal_mag) < 10
    # asserts that past the filtered frequencies the Fourier transform is
    # decreasing.
    assert np.all(np.diff(signal_mag[11:]) < 0.0)

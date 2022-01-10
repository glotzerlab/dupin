import numpy as np
import pytest

from dupin.detect.offline import costs


class BaseCostLinearTest:
    cls = None
    additional_constants = {}

    def test_constants(self):
        assert self.cls.min_size == 3
        assert self.cls._metrics == {"l1", "l2"}
        for attr, value in self.additional_constants.items():
            assert getattr(self.cls, attr) == value

    @pytest.fixture
    def perfect_signal(self):
        N = 50
        # x is implicitly [0, 1] and y is converted to [0, 1]
        s1 = np.linspace(1, 6, N)  # 5 x + 1 -> x
        s2 = np.linspace(5, 0, N)  # -5 x -> -x + 1
        s3 = np.linspace(-7, 13, N)  # 20 x -> x
        self._m = [1, -1, 1]
        self._b = [0, 1, 0]
        self._N = N
        return np.vstack((s1, s2, s3)).T

    def test_get_regression(self, perfect_signal):
        cost_func = self.cls()
        cost_func.fit(perfect_signal)
        m, b = cost_func._get_regression(0, self._N)
        assert np.allclose(m, self._m)
        assert np.allclose(b, self._b)

    def test_error(self, perfect_signal):
        cost_func = self.cls()
        cost_func.fit(perfect_signal)
        assert np.isclose(cost_func.error(0, self._N), 0)
        cost_func = self.cls("l2")
        cost_func.fit(perfect_signal)
        assert np.isclose(cost_func.error(0, self._N), 0)


class TestLinearCost(BaseCostLinearTest):
    cls = costs.CostLinearFit
    additional_constants = {"model": "linear_regression"}

    def test_error_noise(self, rng, perfect_signal):
        noise = rng.normal(
            0, 0.1, size=np.product(perfect_signal.shape)
        ).reshape(perfect_signal.shape)
        noisy_signal = perfect_signal + noise
        cost_func = costs.CostLinearFit()
        cost_func.fit(noisy_signal)
        expected_max_error = np.sum(np.abs(noise))
        assert expected_max_error >= cost_func.error(0, self._N)
        cost_func = costs.CostLinearFit("l2")
        cost_func.fit(noisy_signal)
        expected_max_error = np.sqrt(np.sum(np.square(noise)))
        assert expected_max_error >= cost_func.error(0, self._N)


class TestBiasedCost(BaseCostLinearTest):
    cls = costs.CostLinearBiasedFit
    additional_constants = {"model": "biased_linear_regression"}

    def test_correct_fit(self, rng, perfect_signal):
        # To make a simple test we just set the values to the last one. This
        # preserves the mapping to the unit square without which the regression
        # parameters will change.
        perfect_signal[1:-1] = perfect_signal[-1]
        cost_func = costs.CostLinearBiasedFit()
        cost_func.fit(perfect_signal)
        m, b = cost_func._get_regression(0, self._N)
        assert np.allclose(m, self._m)
        assert np.allclose(b, self._b)

import numpy as np
import pytest
import ruptures as rpt

from dupin.detect.offline import detect


def test_kneedle_elbow_detection():
    x = np.linspace(1e-5, 5, 100)
    y = 1 / x
    elbow = 1
    detected_elbow = detect.kneedle_elbow_detection(y)
    assert np.isclose(detected_elbow, elbow)
    detected_elbow = detect.kneedle_elbow_detection(y, S=2)
    assert np.isclose(detected_elbow, elbow)


def test_two_pass_elbow_detection():
    x = np.linspace(1e-5, 3, 100)
    y = 1 / x
    elbow = 1
    detected_elbow = detect.two_pass_elbow_detection(0.5 * elbow)(y)
    assert np.isclose(detected_elbow, elbow)
    # Should detect next elbow after x=1.
    detected_elbow = detect.two_pass_elbow_detection(1.5 * elbow)(y)
    assert detected_elbow - elbow > 0

    def custom_elbow(costs):
        return 4

    detected_elbow = detect.two_pass_elbow_detection(2, custom_elbow)(y)
    assert detected_elbow == 4


@pytest.mark.parametrize("detector", (rpt.Dynp(), rpt.Binseg(), rpt.BottomUp()))
def test_RupturesWrapper(detector):
    signal, bkps = rpt.pw_constant(50, n_bkps=1)
    wrapper = detect._RupturesWrapper(detector)
    assert wrapper.detector is detector
    points, cost = wrapper(signal, 0)
    assert points == []
    assert cost > 0
    # Test that no error is raised on too many change points (we log this
    # instead.
    signal, bkps = rpt.pw_constant(10, n_bkps=1)
    wrapper(signal, 8)
    # Test correct number of change points are returned, and cost is correct.
    # There is no noise so cost should be roughly zero.
    points, cost = wrapper(signal, 1)
    assert len(points) == 1
    # For some reason the detectors are occasionally off by one. This makes the
    # cost non-zero. The error isn't ours so we test accordingly.
    diff = abs(points[0] - bkps[0])
    assert diff < 2
    if diff == 0:
        assert np.isclose(cost, 0)


class TestSweepDetector:
    @pytest.fixture(params=range(5))
    def construction_kwargs(self, request):
        def update_base_kwargs(**kwargs):
            return {"detector": rpt.Dynp(), "max_change_points": 5, **kwargs}

        def fake_detector(signal, n_change_points):
            return [range(n_change_points), 1 / n_change_points]

        return [
            update_base_kwargs(),
            update_base_kwargs(detector=rpt.Binseg()),
            update_base_kwargs(max_change_points=3),
            update_base_kwargs(
                elbow_detector=detect.two_pass_elbow_detection(1)
            ),
            update_base_kwargs(tolerance=1e-5),
            update_base_kwargs(detector=fake_detector),
        ][request.param]

    def test_construction(self, construction_kwargs):
        detector = detect.SweepDetector(**construction_kwargs)
        if isinstance(construction_kwargs["detector"], rpt.base.BaseEstimator):
            assert isinstance(detector._detector, detect._RupturesWrapper)
            assert (
                detector._detector.detector is construction_kwargs["detector"]
            )
        else:
            assert detector._detector is construction_kwargs["detector"]
        assert (
            detector.max_change_points
            == construction_kwargs["max_change_points"]
        )
        assert detector._elbow_detector is construction_kwargs.get(
            "elbow_detector", detect.kneedle_elbow_detection
        )
        assert detector.tolerance == construction_kwargs.get("tolerance", 1e-3)

    def test_fit(self):
        detector = detect.SweepDetector(
            detector=rpt.Dynp("l1"), max_change_points=7
        )
        signal, bkps = rpt.pw_constant(noise_std=0.1)
        bkps.pop()
        change_points = detector.fit(signal)
        assert len(change_points) == len(bkps)
        assert all(abs(c - b) <= 2 for c, b in zip(change_points, bkps))

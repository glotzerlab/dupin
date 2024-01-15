import numpy as np
import pytest
import ruptures as rpt

from dupin import detect


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

    manual_elbow = 4

    def custom_elbow(costs):
        return manual_elbow

    detected_elbow = detect.two_pass_elbow_detection(2, custom_elbow)(y)
    assert detected_elbow == manual_elbow


@pytest.mark.parametrize("detector", [rpt.Dynp(), rpt.Binseg(), rpt.BottomUp()])
def test_ruptures_wrapper(detector):
    signal, bkps = rpt.pw_constant(50, n_bkps=1)
    wrapper = detect.detect._RupturesWrapper(detector)
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
    max_error = 2
    diff = abs(points[0] - bkps[0])
    assert diff < max_error
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
            assert isinstance(
                detector._detector, detect.detect._RupturesWrapper
            )
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
            detector=rpt.Dynp("l1"), max_change_points=8
        )
        signal, bkps = rpt.pw_constant(noise_std=0.1, delta=(2, 4))
        bkps.pop()
        change_points = detector.fit(signal)
        assert len(change_points) == len(bkps)
        max_error = 2
        assert all(abs(c - b) <= max_error for c, b in zip(change_points, bkps))

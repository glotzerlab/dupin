import numpy as np
import pytest
import ruptures as rpt
import sklearn as sk

import dupin as du


def test_window_iter():
    arr = np.arange(1, 21)
    for i, slice_ in enumerate(du.preprocessing.supervised.window_iter(arr, 3)):
        assert slice_.mean() == i + 2


def test_valid_construction():
    defaults = {
        "loss_function": du.preprocessing.supervised.Window._default_loss,
        "store_intermediate_classifiers": False,
        "n_classifiers": 1,
        "combine_errors": "mean",
    }
    initial_arguments = {
        "classifier": sk.tree.DecisionTreeClassifier(max_depth=1),
        "window_size": 10,
        "test_size": 0.1,
    }

    def check_attrs(init, default):
        window = du.preprocessing.supervised.Window(**init)
        for key in init.keys() | default.keys():
            attr = getattr(window, key)
            expected_attr = init.get(key, default.get(key))
            if attr in (False, True, None):
                assert attr is expected_attr
            else:
                assert attr == expected_attr

    check_attrs(initial_arguments, defaults)
    initial_arguments = {
        "classifier": sk.tree.DecisionTreeClassifier(max_depth=1),
        "window_size": 15,
        "test_size": 0.5,
        "loss_function": lambda cls, x, y: np.sum(y != cls.predict(x)),
        "store_intermediate_classifiers": True,
        "n_classifiers": 10,
        "combine_errors": "median",
    }
    check_attrs(initial_arguments, defaults)


def test_invalid_construction():
    classifier = sk.tree.DecisionTreeClassifier(max_depth=1)
    with pytest.raises(ValueError, match="window_size must be greater than 1."):
        du.preprocessing.supervised.Window(classifier, 1, 0.1)
    with pytest.raises(ValueError, match="test_size must be between 0 and 1."):
        du.preprocessing.supervised.Window(classifier, 2, 0.0)
    with pytest.raises(TypeError):
        du.preprocessing.supervised.Window(classifier, 2, 0.1, {})

    def loss(cls, x, y):
        return np.sum(y != cls.predict(x))

    with pytest.raises(TypeError):
        du.preprocessing.supervised.Window(classifier, 2, 0.1, loss, "foo")
    for n_classifiers in (0, -1):
        with pytest.raises(
            ValueError, match="n_classifiers must be greater than 0."
        ):
            du.preprocessing.supervised.Window(
                classifier, 2, 0.1, loss, True, n_classifiers
            )
    with pytest.raises(
        ValueError, match=r"combine_errors must be in \('mean', 'median'\)\."
    ):
        du.preprocessing.supervised.Window(
            classifier, 2, 0.1, loss, True, 1, "mode"
        )


@pytest.fixture(scope="session", params=(1, 2))
def signal_with_mean_shift(seeds):
    return rpt.pw_constant(100, 2, 1, 0.5, delta=(3, 5), seed=seeds())


@pytest.fixture()
def window():
    classifier = sk.tree.DecisionTreeClassifier(max_depth=1)
    return du.preprocessing.supervised.Window(
        classifier=classifier, window_size=10, test_size=0.5, n_classifiers=20
    )


def test_compute(window, signal_with_mean_shift):
    signal, change_point = signal_with_mean_shift
    error = window.compute(signal)
    max_allowed_error = 6
    assert abs(error.argmin() - change_point[0]) <= max_allowed_error


@pytest.fixture()
def random_signal(rng):
    return rng.uniform(-500_000, 500_000, size=20).reshape((-1, 1))


def test_error_aggregation_methods(window, random_signal):
    assert np.allclose(window._reduce(random_signal), np.mean(random_signal))
    window.combine_errors = "median"
    assert np.allclose(window._reduce(random_signal), np.median(random_signal))


def test_storing_classifiers(window, random_signal):
    window.store_intermediate_classifiers = True
    window.n_classifiers = 1
    window.compute(random_signal)
    expected_len = len(
        list(
            du.preprocessing.supervised.window_iter(
                random_signal, window.window_size
            )
        )
    )
    assert len(window.classifiers_) == expected_len
    assert all(
        isinstance(classifier, sk.tree.DecisionTreeClassifier)
        for window in window.classifiers_
        for classifier in window
    )

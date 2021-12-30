import numpy as np
import pytest

import dupin as du

rng = np.random.default_rng(56564)


def rfloat():
    """Return a random float between -/+ 1,000,000."""
    return 2e6 * (rng.random() - 0.5)


@pytest.fixture
def generator():
    """Return a simple generator function with a, b, c keys."""

    def func():
        return {"a": rfloat(), "b": rfloat(), "c": rfloat()}

    return func


def test_construction(generator):
    """Test valid construction."""
    instance = du.data.SignalAggregator(generator)
    assert instance.logger is None
    assert generator is instance.generator
    logger = du.data.logging.Logger()
    instance = du.data.SignalAggregator(generator, logger)
    assert instance.logger is logger
    assert generator is instance.generator
    assert instance.signals == []


def test_accumulate(generator):
    """Test that accumulate correctly stores data."""
    instance = du.data.SignalAggregator(generator)
    for _ in range(10):
        instance.accumulate()
    assert len(instance.signals) == 10
    assert all(
        all(k in dict_ for k in ("a", "b", "c")) for dict_ in instance.signals
    )


def test_compute():
    """Test compute works with correct iterator."""

    def generator(foo, bar):
        return {"foo": foo, "bar": bar}

    instance = du.data.SignalAggregator(generator)

    def yield_foobar():
        for _ in range(10):
            yield ((), {"foo": rfloat(), "bar": rfloat()})

    instance.compute(yield_foobar())
    assert len(instance.signals) == 10
    assert all(
        all(k in dict_ for k in ("foo", "bar")) for dict_ in instance.signals
    )


def test_to_dataframe(generator):
    """Test correct construction of dataframe."""
    instance = du.data.SignalAggregator(generator)
    for _ in range(10):
        instance.accumulate()
    df = instance.to_dataframe()
    assert df.shape == (10, 3)
    assert np.allclose(
        df.to_numpy(), [list(v.values()) for v in instance.signals]
    )
    assert all(df.columns == ["a", "b", "c"])


def test_logger():
    """Test that data is correctly given to logger."""

    class DummyGenerator(du.data.base.Generator):
        def __call__(self):
            return {"a": np.arange(10)}

    class FakeReducer(du.data.base.DataReducer):
        def compute(self, value):
            self._logger["test"] = True
            return {"": value}

    pipeline = DummyGenerator().pipe(FakeReducer.wraps())
    instance = du.data.SignalAggregator(pipeline, du.data.logging.Logger())
    for _ in range(10):
        instance.accumulate()
    assert len(instance.logger._data) == 10
    for frame_data in instance._logger._data:
        assert frame_data == {"a": {"test": True}}

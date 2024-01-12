import numpy as np
from hypothesis import given
from hypothesis.strategies import (
    data,
    fixed_dictionaries,
    floats,
    integers,
    sets,
    text,
)

import dupin as du


def test_construction():
    """Test valid construction."""

    def generator():
        return {}

    instance = du.data.aggregate.SignalAggregator(generator)
    assert instance.logger is None
    assert generator is instance.generator
    logger = du.data.logging.Logger()
    instance = du.data.aggregate.SignalAggregator(generator, logger)
    assert instance.logger is logger
    assert generator is instance.generator
    assert instance.signals == []


# We set a reasonable cap for test execution
def n_frames(maximum=100):
    return integers(0, maximum)


# Set a reasonable cap on number of data keys
def keys(max_size=50):
    return sets(text(max_size=100), max_size=max_size)


@given(n_frames(), keys(), data())
def test_accumulate(n_frames, keys, data):
    """Test that accumulate correctly stores data."""
    schema = fixed_dictionaries({k: floats() for k in keys})

    def generator():
        return data.draw(schema)

    instance = du.data.aggregate.SignalAggregator(generator)
    for _ in range(n_frames):
        instance.accumulate()
    assert len(instance.signals) == n_frames
    assert all(k in dict_ for k in keys for dict_ in instance.signals)


@given(n_frames(), keys(), data())
def test_compute_no_args(n_frames, keys, data):
    """Test compute works with correct iterator."""
    schema = fixed_dictionaries({k: floats() for k in keys})

    def generator():
        return data.draw(schema)

    def yield_for_generator():
        for _ in range(n_frames):
            yield ((), {})

    instance = du.data.aggregate.SignalAggregator(generator)
    instance.compute(yield_for_generator())
    assert len(instance.signals) == n_frames
    assert all(k in dict_ for k in keys for dict_ in instance.signals)


@given(n_frames(), keys(), data())
def test_compute_with_args(n_frames, keys, data):
    """Test compute works with correct iterator."""

    def generator(**kwargs):
        return {k: kwargs[k] for k in keys}

    instance = du.data.aggregate.SignalAggregator(generator)

    def yield_foobar():
        gen_floats = floats()
        for _ in range(n_frames):
            yield ((), {k: data.draw(gen_floats) for k in keys})

    instance.compute(yield_foobar())
    assert len(instance.signals) == n_frames
    assert all(k in dict_ for k in keys for dict_ in instance.signals)


@given(n_frames(), keys(), data())
def test_to_dataframe(n_frames, keys, data):
    """Test correct construction of dataframe."""
    schema = fixed_dictionaries({k: floats() for k in keys})

    def generator():
        return data.draw(schema)

    instance = du.data.aggregate.SignalAggregator(generator)
    for _ in range(n_frames):
        instance.accumulate()
    df = instance.to_dataframe()
    if n_frames > 0 and len(keys) > 0:
        assert df.shape == (n_frames, len(keys))
        assert np.allclose(
            df.to_numpy(),
            [list(v.values()) for v in instance.signals],
            equal_nan=True,
        )
        assert all(k == c for k, c in zip(df.columns, keys))
        return
    assert len(df.columns) == 0


def test_logger():
    """Test that data is correctly given to logger."""

    class DummyGenerator(du.data.base.Generator):
        def __call__(self):
            return {"a": np.arange(10)}

    class FakeReducer(du.data.base.DataReducer):
        def compute(self, value):
            self._logger["test"] = True
            return {"": value}

    pipeline = DummyGenerator().pipe(FakeReducer())
    instance = du.data.aggregate.SignalAggregator(
        pipeline, du.data.logging.Logger()
    )
    n_frames = 10
    for _ in range(n_frames):
        instance.accumulate()
    assert len(instance.logger._data) == n_frames
    for frame_data in instance._logger._data:
        assert frame_data == {"a": {"test": True}}

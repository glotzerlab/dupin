"""Test dupin.data.logging.

This only test components that are not tested via test_reduce.py or test_map.py.
"""

import dupin as du


def test_frame_mechanism():
    logger = du.data.logging.Logger()
    logger._set_context("foo")
    logger["bar"] = 5
    logger["baz"] = 6
    assert len(logger.frames) == 0
    logger.end_frame()
    assert len(logger.frames) == 1
    assert logger.frames[0] == {"foo": {"bar": 5, "baz": 6}}
    logger._set_context("foo")
    logger["bar"] = 5
    logger._set_context("bar")
    logger["foo"] = 5
    assert len(logger.frames) == 1
    logger.end_frame()
    assert len(logger.frames) == 2
    assert logger.frames[0] == {"foo": {"bar": 5, "baz": 6}}
    assert logger.frames[1] == {"foo": {"bar": 5}, "bar": {"foo": 5}}


def test_to_dataframe(rng):
    logger = du.data.logging.Logger()
    for _ in range(10):
        logger._set_context("foo")
        logger["a"] = rng.random()
        logger["b"] = rng.integers(1000)
        logger._set_context("bar")
        logger["c"] = -rng.random()
        logger["d"] = -rng.integers(1000)
        logger.end_frame()
    df = logger.to_dataframe()
    assert df.shape == (10, 4)
    expected_columns = [("foo", "a"), ("foo", "b"), ("bar", "c"), ("bar", "d")]
    for col, expected_col in zip(df.columns, expected_columns):
        assert col == expected_col
    for i, frame in enumerate(logger.frames):
        index = 0
        for inner_dict in frame.values():
            for value in inner_dict.values():
                assert value == df.iloc[i, index]
                index += 1

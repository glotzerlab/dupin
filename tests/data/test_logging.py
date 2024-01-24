"""Test dupin.data.logging.

This only test components that are not tested via test_reduce.py or test_map.py.
"""

import warnings

import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings

import dupin as du


def not_null(s):
    return "\x00" not in s


def context_dict(values=None):
    if values is None:
        values = st.one_of(st.floats(), st.integers())
    return st.dictionaries(
        st.text(), st.dictionaries(st.text(), values), min_size=1
    )


# TODO: Test duplication of context key behavior
@given(log_contents=st.lists(context_dict()))
def test_frame_mechanism(log_contents):
    logger = du.data.logging.Logger()
    for frame, contexts in enumerate(log_contents):
        drop_keys = []
        was_empty = len(contexts) == 0 or all(
            len(v) == 0 for v in contexts.values()
        )
        for context, data in contexts.items():
            if context in logger._current_frame:
                with pytest.raises(RuntimeError):
                    logger._set_context(context)
                continue
            logger._set_context(context)
            if len(data) == 0:
                drop_keys.append(context)
            for k, v in data.items():
                logger[k] = v
        assert len(logger.frames) == frame
        logger.end_frame()
        assert len(logger.frames) == frame + 1
        if was_empty:
            assert logger.frames[-1] == {}
        for k in drop_keys:
            contexts.pop(k)
        assert contexts == logger.frames[frame]


@st.composite
def orderly_log_content(
    draw,
    value_types=st.sampled_from((st.floats(), st.integers())),
    # Must filter "\x00" to prevent pandas errors.
    keys=st.sets(st.text().filter(not_null), max_size=10),
):
    contexts = st.fixed_dictionaries(
        {
            c: st.fixed_dictionaries(
                {
                    k: draw(value_types)
                    for k in draw(
                        st.sets(
                            st.text().filter(not_null), min_size=1, max_size=5
                        )
                    )
                }
            )
            for c in draw(keys)
        }
    )
    return draw(st.lists(contexts, max_size=10))


def populate_log(logger, log_contents):
    num_empty = 0
    for frame in log_contents:
        if not frame:
            num_empty += 1
        elif all(not v for v in frame.values()):
            num_empty += 1
        for context, data in frame.items():
            logger._set_context(context)
            for k, v in data.items():
                logger[k] = v
        logger.end_frame()
    return num_empty


def check_logger_columns(df, log_contents):
    for frame in log_contents:
        for context, data in frame.items():
            for k in data:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", "indexing past lexsort")
                    expected_col = pd.MultiIndex.from_tuples([(context, k)])[0]
                    assert expected_col in df.columns


def check_logger_values(df, log_contents):
    for i, frame in enumerate(log_contents):
        index = 0
        for inner_dict in frame.values():
            for value in inner_dict.values():
                expected_value = df.iloc[i, index]
                if np.isnan(df.iloc[i, index]):
                    assert np.isnan(value)
                else:
                    assert np.isclose(float(value), expected_value)
                index += 1


@given(orderly_log_content())
@settings(deadline=2_000)  # Allow up to 2 seconds per test.
def test_to_dataframe(log_contents):
    logger = du.data.logging.Logger()
    num_empty = populate_log(logger, log_contents)
    df = logger.to_dataframe()
    # When log contents is empty
    if len(log_contents) == num_empty:
        assert df.shape == (0, 0)
        return
    # Non-empty  log contents
    width = sum(len(v) for v in log_contents[0].values())
    assert df.shape == (len(log_contents), width)
    check_logger_columns(df, log_contents)
    check_logger_values(df, log_contents)

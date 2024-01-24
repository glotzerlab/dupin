import functools
import typing
import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays, floating_dtypes

import dupin as du


@st.composite
def generator_data(
    draw,
    keys=st.sets(st.text(), max_size=50),
    dtype=floating_dtypes(),
    size=st.integers(0, 200),
    allow_nan=True,
):
    array_strat = functools.partial(
        arrays, dtype=dtype, elements={"allow_nan": allow_nan}
    )
    return {k: draw(array_strat(shape=(draw(size),))) for k in draw(keys)}


def _dropnan(arr):
    return arr[~np.isnan(arr)]


def _pairwise(a):
    return zip(a, a[1:])


def add_reducer_tests(cls, validator):
    class TestReducer:
        @pytest.fixture()
        def constructor_args(self):
            return validator.constructor_args

        @pytest.fixture()
        def base_generator(self):
            @du.data.make_generator
            def generator():
                return {}

            return generator

        @given(validator.strategy())
        def test_construction(self, kwargs):
            instance = cls(**kwargs)
            validator.validate_instance(instance, kwargs)

        def test_decorator(self, constructor_args):
            @cls(**constructor_args)
            @du.data.make_generator
            def generate():
                pass

            assert isinstance(generate._generator, du.data.base.CustomGenerator)

            @cls(**constructor_args)
            @du.data.map.Identity()
            @du.data.make_generator
            def generate():
                pass

            assert isinstance(generate._generator, du.data.map.Identity)

        def test_pipeline(self, base_generator, constructor_args):
            pipeline = base_generator.pipe(cls(**constructor_args))
            assert isinstance(pipeline._generator, du.data.base.CustomGenerator)
            pipeline = base_generator.pipe(du.data.map.Identity()).pipe(
                cls(**constructor_args)
            )
            assert isinstance(pipeline._generator, du.data.map.Identity)

        def test_setting_logger(self, base_generator, constructor_args):
            pipeline = base_generator.pipe(du.data.map.Identity()).pipe(
                cls(**constructor_args)
            )
            logger = du.data.logging.Logger()
            pipeline.attach_logger(logger)
            assert pipeline._logger is logger
            assert pipeline._generator._logger is logger
            pipeline.remove_logger()
            assert pipeline._logger is None
            assert pipeline._generator._logger is None

        @given(validator.strategy(), generator_data())
        def test_output(self, kwargs, input_):
            @du.data.make_generator
            def generator():
                return input_

            instance = cls(**kwargs)(generator)
            output = instance()
            validator.validate_output(instance, input_, output)

        @given(validator.strategy(), generator_data())
        @settings(deadline=2e6)
        def test_logging(self, kwargs, input_):
            @du.data.make_generator
            def generator():
                return input_

            logger = du.data.logging.Logger()
            instance = cls(**kwargs)
            pipe = generator.pipe(instance)
            pipe.attach_logger(logger)
            output = pipe()
            logger.end_frame()
            validator.validate_logger(instance, input_, output, logger)

    TestReducer.__name__ = "Test" + cls.__name__

    return TestReducer


class NthGreatestValidator:
    constructor_args: typing.ClassVar[dict[str, typing.Any]] = {
        "indices": [10, -5, 1]
    }

    @st.composite
    @staticmethod
    def strategy(
        draw, indices=st.lists(st.integers(-100, 100), min_size=1, max_size=15)
    ):
        return {"indices": draw(indices)}

    @staticmethod
    def compute_indices(indices):
        fixed_indices = set()
        for i in indices:
            if i == 0:
                fixed_indices.add(-1)
            elif i > 0:
                fixed_indices.add(-i)
            else:
                fixed_indices.add(-i - 1)
        return fixed_indices

    @staticmethod
    def to_positive_index(index, length):
        return index if 0 <= index < length else length + index

    @classmethod
    def validate_instance(cls, instance, kwargs):
        expected_indices = cls.compute_indices(kwargs["indices"])
        assert set(instance._indices) == expected_indices

    @classmethod
    def validate_output(cls, instance, input_, output):
        def check_rank(in_, out_value, index):
            """Assuming a non-nan output check for correct reduction."""
            nth_highest = cls.to_positive_index(index, in_.size)
            greater_than = np.sum(out_value > in_data)
            # Cannot be equal when duplicates in the array exists so check for
            # duplicates
            if greater_than != nth_highest:
                assert np.sum(out_value == in_data) > 1

        def check_index_value(in_, out_value, index):
            """Ensure the output of reducer for an index is as expected."""
            if not du.data.reduce.NthGreatest._fits(in_, index):
                assert np.isnan(out_value)
                return
            # if len(in_) is large enough but too many nans exist we set
            # out[key] to nan. Otherwise we filter out the nan and then talk
            # the nth greatest or least.
            if np.any(np.isnan(in_)):
                filtered_in = in_[~np.isnan(in_)]
                if not du.data.reduce.NthGreatest._fits(filtered_in, index):
                    assert np.isnan(out_value)
                    return
                check_rank(filtered_in, out_value, index)
                return
            check_rank(in_, out_value, index)

        for in_feature, in_data in input_.items():
            if len(in_data) == 0:
                assert in_feature not in output
                continue
            for name, index in zip(instance._names, instance._indices):
                key = "_".join((name, in_feature))
                assert key in output
                check_index_value(in_data, output[key], index)

    @classmethod
    def validate_logger(cls, instance, input_, output, logger):
        def check_index(input_, output_value, log_value, index):
            filtered_in = _dropnan(input_)
            # Not enough non-nan values exist and logger stores nan.
            if not du.data.reduce.NthGreatest._fits(filtered_in, index):
                assert np.isnan(log_value)
            else:
                # Check that the index given to the logger produces the
                # expected output.
                assert output_value == input_[log_value]

        if len(input_) == 0:
            assert logger.frames[0] == {}
            return

        log_data = logger.frames[0]
        for feat, feat_log in log_data.items():
            nth_log = feat_log["NthGreatest"]
            for index in instance._indices:
                index_name, key = cls._get_index_name_key(feat, index)
                assert index_name in nth_log
                check_index(
                    input_[feat], output[key], nth_log[index_name], index
                )

    @staticmethod
    def _get_index_name_key(feature, index):
        index_name = du.data.reduce.NthGreatest._index_name(index)
        key = "_".join((index_name, feature))
        return index_name, key


TestNthGreatest = add_reducer_tests(
    du.data.reduce.NthGreatest, NthGreatestValidator
)


class PercentileValidator:
    constructor_args: typing.ClassVar[dict[str, typing.Any]] = {
        "percentiles": [100, 50, 1]
    }

    @st.composite
    @staticmethod
    def strategy(
        draw, percentiles=st.lists(st.floats(0, 100), min_size=1, max_size=20)
    ):
        return {"percentiles": draw(percentiles | st.none())}

    @classmethod
    def validate_instance(cls, instance, kwargs):
        percentiles = cls._get_percentiles(kwargs["percentiles"])
        assert all(instance._percentiles == np.unique(percentiles))

    @staticmethod
    def _get_percentiles(p):
        return p if p is not None else tuple(i for i in range(0, 101, 10))

    @staticmethod
    def _get_pvalues(arr, percentiles):
        indices = np.asarray(percentiles) / 100.0 * (len(arr) - 1)
        return np.sort(arr)[np.rint(indices).astype(int)]

    @classmethod
    def validate_output(cls, instance, input_, output):
        percentiles = instance._percentiles
        for name, arr in input_.items():
            if len(arr) == 0:
                assert all(f"{p}%_{name}" not in output for p in percentiles)
                continue
            if np.all(np.isnan(arr)):
                assert all(
                    np.isnan(output[f"{p}%_{name}"]) for p in percentiles
                )
                continue
            pvalues = cls._get_pvalues(_dropnan(arr), percentiles)

            for percentile, v in zip(percentiles, pvalues):
                assert output[f"{percentile}%_{name}"] == v

    @classmethod
    def validate_logger(cls, instance, input_, output, logger):
        if len(input_) == 0:
            assert logger.frames[0] == {}
            return

        log_data = logger.frames[0]
        percentiles = instance._percentiles
        for name, arr in input_.items():
            if arr.size == 0:
                assert "Percentile" not in log_data.get(name, {})
                continue
            feat_log = log_data[name]["Percentile"]
            cleaned_arr = _dropnan(arr)
            if cleaned_arr.size == 0:
                for percentile in percentiles:
                    key = f"{percentile}%"
                    assert key in feat_log
                    assert feat_log[key] == 0
                continue
            assert all(a <= b for a, b in _pairwise(percentiles))
            pvalues = cls._get_pvalues(cleaned_arr, percentiles)
            for pv, percentile in zip(pvalues, percentiles):
                key = f"{percentile}%"
                indices = np.flatnonzero(arr == pv)
                assert feat_log[key] in indices


TestPercentile = add_reducer_tests(
    du.data.reduce.Percentile, PercentileValidator
)


class CustomReducerValidator:
    cls: typing.ClassVar[
        du.data.base.DataReducer
    ] = du.data.reduce.CustomReducer
    constructor_args: typing.ClassVar[dict[str, typing.Any]] = {
        "custom_function": lambda d: {"first": d[0]}
    }

    @st.composite
    @staticmethod
    def strategy(
        draw, function=st.sampled_from(("mean", "std", "max", "min", "ptp"))
    ):
        operation = draw(function)
        func = getattr(np, operation)

        def reducer(dist):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                if len(dist) == 0:
                    return {}
                return {operation: func(dist)}

        reducer._func_name = operation

        return {"custom_function": reducer}

    @classmethod
    def validate_instance(cls, instance, kwargs):
        assert instance.function == kwargs["custom_function"]

    @staticmethod
    def validate_output(instance, input_, output):
        func = instance.function
        if len(output) == 0:
            assert all(v.size == 0 for v in input_.values())
            return
        for name, arr in input_.items():
            key = "_".join((func._func_name, name))
            if arr.size == 0:
                assert key not in output
                continue
            assert np.array_equal(
                output[key], next(iter(func(arr).values())), equal_nan=True
            )

    @staticmethod
    def validate_logger(instance, input_, output, logger):
        pass


TestCustomReducer = add_reducer_tests(
    du.data.reduce.CustomReducer, CustomReducerValidator
)


class TeeValidator:
    cls: typing.ClassVar[du.data.base.DataReducer] = du.data.reduce.Tee

    constructor_args: typing.ClassVar[dict[str, typing.Any]] = {
        "reducers": [
            du.data.reduce.NthGreatest([1]),
            du.data.reduce.Percentile([99]),
        ]
    }

    validator_mapping: typing.ClassVar[dict[du.data.base.DataReducer, type]] = {
        du.data.reduce.NthGreatest: NthGreatestValidator,
        du.data.reduce.Percentile: PercentileValidator,
        du.data.reduce.CustomReducer: CustomReducerValidator,
    }

    @st.composite
    @staticmethod
    def strategy(draw, size=st.integers(1, 10)):
        cls_choice = st.sampled_from(
            [
                (du.data.reduce.NthGreatest, NthGreatestValidator.strategy()),
                (du.data.reduce.Percentile, PercentileValidator.strategy()),
                (
                    du.data.reduce.CustomReducer,
                    CustomReducerValidator.strategy(),
                ),
            ]
        )
        reducers = []
        for _ in range(draw(size)):
            cls, kwargs_strat = draw(cls_choice)
            reducers.append(cls(**draw(kwargs_strat)))

        return {"reducers": reducers}

    @classmethod
    def validate_instance(cls, instance, kwargs):
        for i, pipeline_obj in enumerate(kwargs["reducers"]):
            assert instance._reducers[i] is pipeline_obj

    @classmethod
    def validate_output(cls, instance, input_, output):
        for reducer in instance._reducers:
            cls.validator_mapping[type(reducer)].validate_output(
                reducer, input_, output
            )

    @classmethod
    def validate_logger(cls, instance, input_, output, logger):
        if len(input_) == 0:
            assert logger.frames[0] == {}
            return
        encountered_types = set()
        for reducer in reversed(instance._reducers):
            # Logger will override the same key so we only look at the last of
            # a type.
            if type(reducer) in encountered_types:
                continue
            encountered_types.add(type(reducer))
            cls.validator_mapping[type(reducer)].validate_logger(
                reducer, input_, output, logger
            )


TestTee = add_reducer_tests(du.data.reduce.Tee, TeeValidator)

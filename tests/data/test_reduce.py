import inspect

import numpy as np
import pytest

import dupin as du


class BaseReducerTest:
    cls = None

    @pytest.fixture
    def generator(self):
        return du.data.base.CustomGenerator(lambda: {"a": np.arange(100)})

    @pytest.fixture(params=(0, 1, 2))
    def n(self, request):
        """Fixture to allow for other parameterized fixtures."""
        return request.param

    @pytest.fixture
    def valid_spec(self, n):
        """Return the nth valid spec."""
        raise NotImplementedError

    def test_decorator(self, valid_spec):
        @self.cls.wraps(**valid_spec)
        def generate():
            pass

        assert inspect.isfunction(generate._generator)

        @self.cls.wraps(**valid_spec)
        @du.data.map.Identity.wraps()
        def generate():
            pass

        assert isinstance(generate._generator, du.data.map.Identity)

    def test_pipeline(self, generator, valid_spec):
        pipeline = generator.pipe(self.cls.wraps(**valid_spec))
        assert isinstance(pipeline._generator, du.data.base.CustomGenerator)
        pipeline = generator.pipe(du.data.map.Identity.wraps()).pipe(
            self.cls.wraps(**valid_spec)
        )
        assert isinstance(pipeline._generator, du.data.map.Identity)

    def test_setting_logger(self, generator, valid_spec):
        pipeline = generator.pipe(du.data.map.Identity.wraps()).pipe(
            self.cls.wraps(**valid_spec)
        )
        logger = du.data.logging.Logger()
        pipeline.attach_logger(logger)
        assert pipeline._logger is logger
        assert pipeline._generator._logger is logger
        pipeline.remove_logger()
        assert pipeline._logger is None
        assert pipeline._generator._logger is None

    def test_output(self, generator, valid_spec):
        instance = self.cls(generator, **valid_spec)
        output = instance()
        self.validate_output(output, instance)


class TestNthGreatest(BaseReducerTest):
    cls = du.data.reduce.NthGreatest

    @pytest.fixture
    def valid_spec(self, n):
        return (
            {"indices": [1, -1]},
            {"indices": [10, 5, 1]},
            {"indices": [-10, -5, -3]},
        )[n]

    @staticmethod
    def compute_indices(indices):
        return {-i if i >= 1 else abs(i + 1) for i in indices}

    @staticmethod
    def to_positive_index(index, length):
        return index if 0 <= index < length else length + index

    def test_construction(self, generator, valid_spec):
        instance = self.cls(generator, **valid_spec)
        assert instance._generator is generator
        expected_indices = self.compute_indices(valid_spec["indices"])
        assert set(instance._indices) == expected_indices

    @staticmethod
    def validate_output(output, reducer):
        for index in reducer._indices:
            index_name = reducer._index_name(index)
            key = "_".join((index_name, "a"))
            assert TestNthGreatest.to_positive_index(index, 100) == output[key]

    def test_logging(self, generator, valid_spec):
        logger = du.data.logging.Logger()
        instance = self.cls(generator, **valid_spec)
        instance.attach_logger(logger)
        instance()
        logger.end_frame()
        logger_data = logger.frames[0]["a"]["NthGreatest"]
        for index in instance._indices:
            index_name = instance._index_name(index)
            assert self.to_positive_index(index, 100) == logger_data[index_name]


class TestPercentile(BaseReducerTest):
    cls = du.data.reduce.Percentile

    @pytest.fixture(params=(0, 1, 2, 3))
    def n(self, request):
        """Fixture to allow for other parameterized fixtures."""
        return request.param

    @pytest.fixture
    def valid_spec(self, n):
        return (
            {},
            {"percentiles": [1, 99]},
            {"percentiles": [10, 5, 1]},
            {"percentiles": [100, 50, 1]},
        )[n]

    def test_construction(self, generator, valid_spec):
        instance = self.cls(generator, **valid_spec)
        assert instance._generator is generator
        if valid_spec != {}:
            assert instance._percentiles == valid_spec["percentiles"]

    @staticmethod
    def get_index(percent, length):
        return np.round(float(percent) / 100 * (length - 1)).astype(int).item()

    @staticmethod
    def validate_output(output, reducer):
        for percentile in reducer._percentiles:
            key = f"{percentile}%_a"
            assert TestPercentile.get_index(percentile, 100) == output[key]

    def test_logging(self, generator, valid_spec):
        logger = du.data.logging.Logger()
        instance = self.cls(generator, **valid_spec)
        instance.attach_logger(logger)
        instance()
        logger.end_frame()
        logger_data = logger.frames[0]["a"]["Percentile"]
        for percentile in instance._percentiles:
            key = f"{percentile}%"
            assert self.get_index(percentile, 100) == logger_data[key]


class TestTee(BaseReducerTest):
    cls = du.data.reduce.Tee

    @pytest.fixture
    def n(self):
        return 0

    @pytest.fixture
    def valid_spec(self, n):
        return {
            "reducers": [
                du.data.reduce.NthGreatest.wraps([1]),
                du.data.reduce.Percentile.wraps([99]),
            ]
        }

    def test_construction(self, generator, valid_spec):
        instance = self.cls(generator, **valid_spec)
        assert instance._generator is generator
        for i, pipeline_obj in enumerate(valid_spec["reducers"]):
            assert isinstance(instance._reducers[i], pipeline_obj._target_cls)

    @staticmethod
    def validate_output(output, reducer):
        TestNthGreatest.validate_output(output, reducer._reducers[0])
        TestPercentile.validate_output(output, reducer._reducers[1])

    def test_logging(self, generator, valid_spec):
        logger = du.data.logging.Logger()
        instance = self.cls(generator, **valid_spec)
        instance.attach_logger(logger)
        instance()
        logger.end_frame()
        assert "Percentile" in logger.frames[0]["a"]
        assert "NthGreatest" in logger.frames[0]["a"]


class TestCustomReducer(BaseReducerTest):
    cls = du.data.reduce.CustomReducer

    @pytest.fixture
    def valid_spec(self, n):
        return (
            {"custom_function": lambda arr: {"40": arr[40]}},
            {"custom_function": lambda arr: {"35": arr[35]}},
            {"custom_function": lambda arr: {"62": arr[62]}},
        )[n]

    def test_construction(self, generator, valid_spec):
        instance = self.cls(generator, **valid_spec)
        assert instance._generator is generator
        assert instance.function == valid_spec["custom_function"]

    @staticmethod
    def validate_output(output, reducer):
        key, value = output.popitem()
        index = int(key[:2])
        assert value == index

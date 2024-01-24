import numpy as np
import pytest

import dupin as du


class TestCustomGenerator:
    @pytest.fixture(params=range(3))
    def n(self, request):
        return request.param

    @pytest.fixture()
    def function(self, n):
        if n == 0:

            def func():
                return {"foo": np.arange(10)}

        elif n == 1:

            def func(a):
                return {"foo": np.arange(a)}

        else:

            def func(a, b):
                return {"bar": b - a}

        return func

    @pytest.fixture()
    def generator(self, function):
        return du.data.base.CustomGenerator(function)

    @pytest.fixture(params=range(3))
    def valid_call(self, n, request):
        return [
            ((), {}),
            ((), {}),
            ((), {}),
            ((1,), {}),
            ((-1,), {}),
            ((), {"a": 5}),
            ((1,), {"b": 2}),
            ((-1, -3), {}),
            ((), {"a": 5, "b": 15}),
        ][n * 3 + request.param]

    def test_call(self, generator, valid_call):
        through_generator = generator(*valid_call[0], **valid_call[1])
        through_func = generator.function(*valid_call[0], **valid_call[1])
        assert through_func.keys() == through_generator.keys()
        for k, v in through_generator.items():
            assert np.allclose(through_func[k], v)

    def test_piping(self, generator):
        pipeline = generator.pipe(du.data.reduce.Percentile())
        assert isinstance(pipeline, du.data.reduce.Percentile)
        assert isinstance(pipeline._generator, du.data.base.CustomGenerator)

    def test_map(self, generator):
        pipeline = generator.map(du.data.map.map_(lambda: None))
        assert isinstance(pipeline, du.data.map.CustomMap)
        assert isinstance(pipeline._generator, du.data.base.CustomGenerator)

    def test_reduce(self, generator):
        pipeline = generator.reduce(du.data.reduce.Percentile())
        assert isinstance(pipeline, du.data.reduce.Percentile)
        assert isinstance(pipeline._generator, du.data.base.CustomGenerator)

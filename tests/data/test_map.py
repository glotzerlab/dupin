"""Test dupin.data.map."""

import inspect

import freud
import numpy as np
import pytest

import dupin as du

SPH_HARM_NUMBER = [2, 3, 4]


class BaseMapTest:
    cls = None

    @pytest.fixture()
    def generator(self):
        return du.data.freud.FreudDescriptor(
            freud.order.Steinhardt(l=SPH_HARM_NUMBER),
            {"particle_order": [str(sph_harm) for sph_harm in SPH_HARM_NUMBER]},
        )

    @pytest.fixture()
    def valid_spec(self):  # noqa: PT004
        """Return a valid spec."""
        raise NotImplementedError

    def test_decorator(self, valid_spec):
        @du.data.reduce.NthGreatest([1])
        @self.cls(**valid_spec())
        def generate():
            pass

        assert inspect.isfunction(generate._generator._generator)

        @du.data.reduce.NthGreatest([1])
        @self.cls(**valid_spec())
        @du.data.map.Identity()
        def generate():
            pass

        assert isinstance(generate._generator._generator, du.data.map.Identity)

    def test_pipeline(self, generator, valid_spec):
        pipeline = generator.pipe(self.cls(**valid_spec())).pipe(
            du.data.reduce.NthGreatest([1])
        )
        assert isinstance(
            pipeline._generator._generator, du.data.freud.FreudDescriptor
        )
        pipeline = (
            generator.pipe(du.data.map.Identity())
            .pipe(self.cls(**valid_spec()))
            .pipe(du.data.reduce.NthGreatest([1]))
        )
        assert isinstance(pipeline._generator._generator, du.data.map.Identity)

    def test_setting_logger(self, generator, valid_spec):
        pipeline = generator.pipe(du.data.map.Identity()).pipe(
            self.cls(**valid_spec())
        )
        logger = du.data.logging.Logger()
        pipeline.attach_logger(logger)
        assert pipeline._logger is logger
        assert pipeline._generator._logger is logger
        pipeline.remove_logger()
        assert pipeline._logger is None
        assert pipeline._generator._logger is None

    def test_output(self, generator, valid_spec, mock_fcc_system):
        """Test the map outputs the expected values."""
        instance = generator.pipe(self.cls(**valid_spec()))
        box, positions = mock_fcc_system(noise=1e-2)
        nlist = (
            freud.locality.AABBQuery(box, positions)
            .query(positions, {"num_neighbors": 12, "exclude_ii": True})
            .toNeighborList()
        )
        output = instance((box, positions), neighbors=nlist)
        self.validate_output(
            output,
            instance._generator.compute.particle_order,
            {"system": (box, positions), "neighbors": nlist},
        )

    @staticmethod
    def validate_output(output, compute_arr, passed_args):
        raise NotImplementedError


class TestIdentity(BaseMapTest):
    cls = du.data.map.Identity

    @pytest.fixture()
    def valid_spec(self):
        return lambda: {}

    @staticmethod
    def validate_output(output, compute_arr, passed_args):
        for i, arr in enumerate(output.values()):
            assert np.all(arr == compute_arr[:, i])


class TestSpatialAveraging(BaseMapTest):
    cls = du.data.spatial.NeighborAveraging

    @pytest.fixture()
    def valid_spec(self):
        return lambda: {"expected_kwarg": "neighbors", "remove_kwarg": False}

    @staticmethod
    def compute_average(arr, neighbors):
        # method assumes that exclude_ii == True
        averaged_arr = np.copy(arr)
        counts = np.ones(len(arr))
        for i, j in neighbors:
            averaged_arr[i] += arr[j]
            counts[i] += 1
        return averaged_arr / counts[:, None]

    @staticmethod
    def validate_output(output, compute_arr, passed_args):
        averaged_arr = TestSpatialAveraging.compute_average(
            compute_arr, passed_args["neighbors"]
        )
        for i, arr in enumerate(output.values()):
            assert np.allclose(arr, averaged_arr[:, i])


class TestTee(BaseMapTest):
    cls = du.data.map.Tee

    @pytest.fixture()
    def valid_spec(self):
        def spec():
            return {
                "maps": [
                    du.data.map.Identity(),
                    du.data.spatial.NeighborAveraging("neighbors", False),
                ]
            }

        return spec

    @staticmethod
    def validate_output(output, compute_arr, passed_args):
        identity_dict = {
            k: v for k, v in output.items() if not k.startswith("spa")
        }
        assert len(identity_dict) == len(SPH_HARM_NUMBER)
        TestIdentity.validate_output(identity_dict, compute_arr, passed_args)
        spatial_dict = {k: v for k, v in output.items() if k.startswith("spa")}
        assert len(spatial_dict) == len(SPH_HARM_NUMBER)
        TestSpatialAveraging.validate_output(
            spatial_dict, compute_arr, passed_args
        )


class TestCustomMap(BaseMapTest):
    cls = du.data.base.CustomMap

    @pytest.fixture()
    def valid_spec(self):
        def double(arr):
            return {"doubled": arr * 2}

        def spec():
            return {"custom_function": double}

        return spec

    @staticmethod
    def validate_output(output, compute_arr, passed_args):
        for i, value in enumerate(output.values()):
            assert np.allclose(value, 2 * compute_arr[:, i])

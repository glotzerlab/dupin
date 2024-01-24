"""Test the dupin.data.freud module."""

import freud
import numpy as np
import pytest

import dupin as du


def specs():
    """Yield valid constructor arguments for FreudDescriptor."""
    yield {
        "compute": freud.density.LocalDensity(r_max=3, diameter=1),
        "attrs": "density",
    }
    yield {
        "compute": freud.order.Steinhardt(l=list(range(10))),
        "attrs": {"particle_order": [f"Q_{_l}" for _l in range(10)]},
    }
    yield {
        "compute": freud.locality.Voronoi(),
        "attrs": ["volumes"],
        "compute_method": "compute",
    }


@pytest.fixture(params=specs())
def spec(request):
    """Return individual specifications."""
    return request.param


def test_construction(spec):
    """Test that parameters are set correctly at construction."""

    def expected_attrs(attrs):
        if isinstance(attrs, str):
            return {attrs: attrs}
        if isinstance(attrs, list):
            return {attr: attr for attr in attrs}
        return {k: k if v is None else v for k, v in attrs.items()}

    instance = du.data.freud.FreudDescriptor(**spec)
    assert instance.compute is spec["compute"]
    assert instance.compute_method == spec.get("compute_method", "compute")
    assert instance.attrs == expected_attrs(spec["attrs"])


def invalid_specs():
    yield {"compute": None, "attrs": ["positions"]}
    yield {"compute": freud.locality.Voronoi(), "attrs": 5}
    yield {
        "compute": freud.locality.Voronoi(),
        "attrs": "volumes",
        "compute_name": "foo",
    }
    yield {
        "compute": freud.locality.Voronoi(),
        "attrs": "volumes",
        "compute_name": None,
    }
    yield {
        "compute": freud.locality.Voronoi(),
        "attrs": "volumes",
        "compute_name": 5,
    }


@pytest.mark.parametrize("invalid_spec", invalid_specs())
def test_invalid_specifications(invalid_spec):
    """Test that invalid values fail construction."""
    with pytest.raises((ValueError, TypeError)):
        du.data.freud.FreudDescriptor(**invalid_spec)


def test_invalid_calls(mock_random_system, spec):
    """Test that calls for ill formed objects or with bad arguments fail."""
    instance = du.data.freud.FreudDescriptor(**spec)
    with pytest.raises(TypeError):
        instance()
    if isinstance(instance.compute, freud.order.Steinhardt):
        with pytest.raises(NotImplementedError):
            instance(mock_random_system())
    elif isinstance(instance.compute, freud.locality.Voronoi):
        with pytest.raises(TypeError):
            instance(mock_random_system(), neighbors={"num_neighbors": 6.0})


def test_call(mock_random_system, spec):
    """Test the call operator returns properly named and unaduterated data."""
    instance = du.data.freud.FreudDescriptor(**spec)
    if isinstance(instance.compute, freud.locality._PairCompute):
        output = instance(
            mock_random_system(),
            neighbors={"num_neighbors": 12, "exclude_ii": True},
        )
    else:
        output = instance(mock_random_system())
    for attr, keys in instance.attrs.items():
        compute_values = getattr(instance.compute, attr)
        if isinstance(keys, list):
            # Expect 2 arrays when multiple names for a compute given
            assert compute_values.ndim == 2  # noqa: PLR2004
            assert len(keys) == compute_values.shape[1]
            for i, key in enumerate(keys):
                assert np.allclose(compute_values[:, i], output[key])
            continue
        # Handle single feature case
        key = attr if keys is None else keys
        assert np.allclose(output[key], compute_values)

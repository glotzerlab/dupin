"""Provide fixture and other testing helpers."""

import freud
import numpy as np
import pytest


@pytest.fixture(scope="session")
def rng():  # noqa: D401
    """A random number generator for tests that have need for random numbers."""
    return np.random.default_rng(5645646)


@pytest.fixture(scope="session")
def mock_fcc_system(rng):
    """Create 3D FCC systems.

    Parameters:
        ns: tuple[int, int, int], optional
            Number of unit cells in each dimension. Defaults to ``(2, 2, 2)``.
        a: float, optional
            The lattice spacing. Defaults to 1.0.
        noise: float, optional
            The standard deviation of noise about the lattice sites. Defaults to
            ``1e-3``.
    """

    def system_factory(ns=(2, 2, 2), a=1.0, noise=1e-3):  # noqa: E741
        return freud.data.UnitCell.fcc().generate_system(
            ns, a, noise, rng.integers(1e4)
        )

    return system_factory


@pytest.fixture(scope="session")
def mock_random_system():
    """Create 2 or 3D random position systems.

    Parameters:
        N: int, optional
            Number of particles. Defaults to 100.
        l: float, optional
            The box length. Defaults to 10.0.
        dimensions: int, optional
            The dimension of the system. Defaults to 3.
    """

    def system_factory(N=50, l=10.0, dimensions=3):  # noqa: E741
        return freud.data.make_random_system(
            box_size=l, num_points=N, is2D=dimensions == 2
        )

    return system_factory

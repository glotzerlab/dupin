"""Miscellaneous functions and functions to avoid circular imports."""

from typing import Tuple


def _str_isinstance(instance: object, cls_strings: Tuple[str, ...]) -> bool:
    cls_name = ".".join((instance.__module__, instance.__class__.__name__))
    return any(cls_name in cls_str for cls_str in cls_strings)


def _state_to_freud_system(state):
    if isinstance(state, tuple):
        return state
    if _str_isinstance(state, "hoomd.state.State"):
        state = state.snapshot
    hoomd_snapshot_classes = (
        "hoomd.Snapshot",
        "hoomd.data.local_access.LocalSnapshot",
    )
    gsd_classes = ("gsd.hoomd.Snapshot",)
    if _str_isinstance(state, hoomd_snapshot_classes + gsd_classes):
        return (state.configuration.box, state.particles.position)
    else:
        raise TypeError("state is not a valid type.")


def _state_to_id(state):
    hoomd_snapshot_classes = (
        "hoomd.Snapshot",
        "hoomd.data.local_access.LocalSnapshot",
    )
    gsd_classes = ("gsd.hoomd.Snapshot",)
    if isinstance(state, tuple) or _str_isinstance(
        state, hoomd_snapshot_classes + gsd_classes
    ):
        return id(state)
    if _str_isinstance(state, "hoomd.state.State"):
        return state._simulation.timestep
    else:
        raise TypeError("state is not a valid type.")

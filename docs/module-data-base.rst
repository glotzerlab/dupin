dupin.data.base
-------------------------

.. rubric:: Overview

.. py:currentmodule:: dupin.data.base

.. autosummary::
    :nosignatures:

    CustomGenerator
    DataMap
    DataModifier
    DataReducer
    Generator
    GeneratorLike
    PipeComponent

.. rubric:: Details

.. automodule:: dupin.data.base
    :synopsis: Base classes for data generation and manipulation.
    :members: CustomGenerator,
        DataMap,
        DataModifier,
        DataReducer,
        Generator,
        GeneratorLike,
        PipeComponent


.. class:: GeneratorLike
    A type hint for objects that act like data generators for dupin.

    The object can either be a `Generator`, `DataMap`, or callable with the
    appropriate return value.

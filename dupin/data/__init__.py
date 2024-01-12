"""Classes and functions for generating and recording system descriptors.

Data Model
==========

Steps
-----
Data creation and modification occurs in three steps:
    1. Generation - create initial features from trajectory data.
    2. Mapping - transform initial features into new distributions (e.g. spatial
       averaging).
    3. Reduction - reduce distributions (arrays) into (multiple) single
       features.

Data Format
-----------
``dupin`` uses a pipeline approach to generate and modify data from raw
trajectory data. Throughout this pipeline data is stored in dictionaries where
the keys denote the feature stored and the value is the actual feature or array
(distribution). At each step in the pipeline past generation, a pipeline
component is expected to not only return the modified data but also string keys
referring to the modification. For example, for a reducer that takes the maximum
a propery return value of `DataModifier.compute` would be
:literal:`{"max": np.max(distribuition)}`. These keys get concatenated with the
previous key so that if the key for the distribution in the previous examples
was ``"density"`` the new dictionary key would be ``"max_density"``.

Pipeline Creation
-----------------
dupin supports two distinct methods for creating pipelines: the builder
style and decorator style. While functionally equivalent the two styles read
quite different.

Builder Method
++++++++++++++
The builder approach uses the `PipeComponent.pipe`, `PipeComponent.map`,
and `PipeComponent.reduce` methods to create a pipeline. The pipeline reads
left to right.

Example::

    generator.map(lambda x: {"mag": np.floor(np.log10(x))}
    ).map(lambda x: {"clipped": np.clip(x, 2)}
    ).reduce(lambda x: {"min": np.min(x)})

Decorator Method
++++++++++++++++
The decorator syntax uses Python's decorator syntax and explicit function
composition to construct the pipeline. Consequently, the pipeline reads right to
left and bottom to top.

Example::

    @du.data.reduce_(lambda x: {"min": np.min(x)})
    @du.data.map_(lambda x: {"clipped": np.clip(x, 2)})
    @du.data.map_(lambda x: {"mag": np.floor(np.log10(x))})
    @du.data.make_generator
    def generator():
        pass

"""
from . import aggregate, base, freud, logging, map, reduce, spatial
from .base import make_generator

__all__ = (
    "aggregate",
    "base",
    "freud",
    "logging",
    "map",
    "reduce",
    "spatial",
    "make_generator",
)

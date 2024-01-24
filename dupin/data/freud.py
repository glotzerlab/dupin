"""Interface from `freud` to dupin.

freud is a Python package for analyzing molecular trajectory data.

The interface consist of a means of creating generators from freud computes.
This implementation is not privledged and one could implement a generator from
freud computes independent of this module.

Note
----
    Requires the `freud` package.

"""

from collections.abc import Sequence
from typing import Any, Union

import numpy.typing as npt

import dupin.errors as error

from . import base

# optional dependencies
try:
    import freud
except ImportError:
    freud = error._RaiseModuleError("freud")


class FreudDescriptor(base.Generator):
    """Provides an interface for using freud computes in dupin.

    The generator produces a dictionary of compute attributes selected by the
    user. The feature identifiers (i.e. keys) are provided by the ``attrs``
    constructor argument, and are the attribute names unless explicitly
    specified otherwise. See the constructor documentation for more information.

    The generator wraps the compute method of the freud compute and takes the
    same signature for ``FreudDescriptor()``.

    Parameters
    ----------
    compute:
        A freud object for computation.
    attrs: str or Sequence[str] or dict[str, str]
        If a string, the name of the attribute to use. This will also be the
        key name. If a sequence (list, tuple, etc) of string, all strings
        are considered attributes to use, and the string is used as the
        feature key as well. If a dictionary, the keys are the names of the
        attributes in the freud compute, and the values are the identifiers
        to use in dupin (``None`` can be used as a proxy for the same as the
        attribute name). For two dimensional arrays, a list of strings is
        expected in place of a single string where each column is an
        independent feature.
    compute_method: `str`, optional
        The method name to use for computing the attrs specified. Defaults
        to "compute".

    Note
    ----
        freud in some cases lumps multiple features into a two dimensional array
        where the rows are individual particles and columns are features.
        `FreudDescriptor` supports these features by using a list of string
        feature names in the constructor argument ``attrs``.
    """

    _prepend_compute_docstring = "The composed docstring is below."

    def __init__(
        self,
        compute: "freud.util._Compute",
        attrs: Union[str, list[str], dict[str, str]],
        compute_method: str = "compute",
    ) -> None:
        if not hasattr(compute, compute_method):
            msg = "compute_method must exist for the compute."
            raise ValueError(msg)
        self.compute = compute
        if isinstance(attrs, str):
            attrs = {attrs: attrs}
        elif isinstance(attrs, Sequence):
            attrs = {attr: attr for attr in attrs}
        self.attrs = self._convert_attrs(attrs)
        self.compute_method = compute_method

    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> dict[str, Union[float, npt.ArrayLike]]:
        """Return computed attributes specified in a dictionary.

        The keys of the dictionary are the attributes specified, unless a `dict`
        was passed for ``attrs``, then then the keys are the values of that
        `dict`.
        """
        getattr(self.compute, self.compute_method)(*args, **kwargs)
        collected_data = {}
        for attr in self.attrs:
            # If their are multiple names for an attribute assume attribute is
            # multidimensional and split along dimension.
            datum = getattr(self.compute, attr)
            name = self.attrs[attr]
            if not isinstance(name, str) and isinstance(name, Sequence):
                if len(datum.shape) != 2:  # noqa: PLR2004
                    msg = (
                        f"Cannot specify multiple names for a single dimension "
                        f"array. Problem attr {attr}"
                    )
                    raise ValueError(msg)
                if datum.shape[1] != len(name):
                    msg = (
                        f"The correct number of names not specified for attr "
                        f"{attr}. Expected {datum.shape[1]} got {len(name)}."
                    )
                    raise ValueError(msg)
                collected_data.update({n: d for n, d in zip(name, datum.T)})
            else:
                collected_data[name] = datum
        return collected_data

    def _convert_attrs(self, attrs):
        if isinstance(attrs, str):
            return {attrs: attrs}
        if isinstance(attrs, Sequence):
            return {attr: attr for attr in attrs}
        if isinstance(attrs, dict):
            if not all(
                isinstance(v, (type(None), str, Sequence))
                for v in attrs.values()
            ):
                msg = "Improper specification of compute attributes"
                raise ValueError(msg)
            return attrs
        msg = (
            "attrs must be a str, list of strings, or dict of string values "
            "and keys"
        )
        raise TypeError(msg)

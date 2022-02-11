"""Interface from freud to dupin."""

from collections.abc import Sequence
from typing import Any, Dict, List, Union

import numpy.typing as npt

import dupin.errors as error

from . import base

# optional dependencies
try:
    import freud
except ImportError:
    freud = error._RaiseModuleError("freud")


class FreudDescriptor(base.Generator):
    """Defines the interface between freud and dupin."""

    _prepend_compute_docstring = "The composed docstring is below."

    def __init__(
        self,
        compute: "freud.util._Compute",
        attrs: Union[str, List[str]],
        compute_method: str = "compute",
    ) -> None:
        """Create a `FreudDescriptor` object.

        Parameters
        ----------
        compute:
            A freud object for computation.
        attrs: str or Sequence[str] or dict[str, str]
            A mapping of attribute names to desired signal names. If the value
            in a entry is ``None`` the key value is used. A single string or
            sequence of strings can be passed and will be converted to the
            appropriate dict instance.
        compute_method: `str`, optional
            The method name to use for computing the attrs specified. Defaults
            to "compute".
        """
        if not hasattr(compute, compute_method):
            raise ValueError("compute_method must exist for the compute.")
        self.compute = compute
        if isinstance(attrs, str):
            attrs = {attrs: attrs}
        elif isinstance(attrs, Sequence):
            attrs = {attr: attr for attr in attrs}
        self.attrs = self._convert_attrs(attrs)
        self.compute_method = compute_method

    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> Dict[str, Union[float, npt.ArrayLike]]:
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
                if len(datum.shape) != 2:
                    raise ValueError(
                        f"Cannot specify multiple names for a single dimension "
                        f"array. Problem attr {attr}"
                    )
                if datum.shape[1] != len(name):
                    raise ValueError(
                        f"The correct number of names not specified for attr "
                        f"{attr}. Expected {datum.shape[1]} got {len(name)}."
                    )
                collected_data.update({n: d for n, d in zip(name, datum.T)})
            else:
                collected_data[name] = datum
        return collected_data

    def _convert_attrs(self, attrs):
        if isinstance(attrs, str):
            return {attrs: attrs}
        elif isinstance(attrs, Sequence):
            return {attr: attr for attr in attrs}
        elif isinstance(attrs, dict):
            if not all(
                isinstance(v, (type(None), str, Sequence))
                for v in attrs.values()
            ):
                raise ValueError("Improper specification of compute attributes")
            return attrs
        raise TypeError(
            "attrs must be a str, list of strings, or dict of string values "
            "and keys"
        )

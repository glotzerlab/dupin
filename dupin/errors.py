"""Base error classes or associated classes."""


class _RaiseModuleError:
    """Raises a module error when a non-available feature is used.

    This is necessary to allow for optional dependencies with default imports
    and useful error messages.
    """

    def __init__(self, module):
        self.module = module

    def __getattribute__(self, attr):
        raise ModuleNotAvailableError(
            f"The {attr} feature is not available as the module {self.module} "
            f"is not available."
        )


class ModuleNotAvailableError(ImportError):
    """Raise when a feature requires an unavailable feature."""

    pass

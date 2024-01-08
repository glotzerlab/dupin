"""Base error classes or associated classes."""


class _RaiseModuleError:
    """Raises a module error when a non-available feature is used.

    This is necessary to allow for optional dependencies with default imports
    and useful error messages.
    """

    def __init__(self, module):
        self.module = module

    def __getattribute__(self, attr):
        msg = (
            f"The {attr} feature is not available as the module {self.module} "
            f"is not available."
        )
        raise ModuleNotAvailableError(msg)


class ModuleNotAvailableError(ImportError):
    """Denotes when a module requires an unavailable package."""

    pass

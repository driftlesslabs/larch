from __future__ import annotations

import warnings

import xarray as xr


class NameConflictWarning(Warning):
    """Warning for conflicts in name registration."""


def _register_classmethod(name, cls):
    def decorator(classmeth):
        if hasattr(cls, name):
            warnings.warn(
                f"registration of classmethod {classmeth!r} under name {name!r} "
                f"for type {cls!r} is overriding a preexisting attribute with "
                f"the same name.",
                NameConflictWarning,
                stacklevel=2,
            )
        setattr(cls, name, classmethod(classmeth))
        return classmeth

    return decorator


def register_dataarray_classmethod(func):
    """
    Register a custom classmethod on xarray.DataArray objects.

    Use this as a decorator to add class methods.

    Parameters
    ----------
    func : Callable
        Class method to add.  The name is inferred from the
        original name of this function.
    """
    return _register_classmethod(func.__name__, xr.DataArray)(func)


def register_dataset_classmethod(func):
    """
    Register a custom classmethod on xarray.Dataset objects.

    Use this as a decorator to add class methods.

    Parameters
    ----------
    func : Callable
        Class method to add.  The name is inferred from the
        original name of this function.
    """
    return _register_classmethod(func.__name__, xr.Dataset)(func)

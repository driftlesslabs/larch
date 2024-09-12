from __future__ import annotations

import contextlib
import warnings


@contextlib.contextmanager
def ignore_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


class LarchWarning(Warning):
    """Base class for all Larch warnings."""


class NoDefaultValueWarning(LarchWarning):
    pass


class WeightRescaleWarning(Warning):
    """Warning issued when weights are rescaled."""


def no_default_value(message):
    # warnings.warn(message, NoDefaultValueWarning, stacklevel=2)
    pass

from __future__ import annotations

from collections.abc import MutableMapping
from copy import copy
from types import SimpleNamespace


class Namespace(SimpleNamespace, MutableMapping):
    def __getitem__(self, item):
        try:
            return getattr(self, item)
        except AttributeError:
            raise KeyError(item) from None

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __delitem__(self, key):
        try:
            delattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def keys(self):
        return self.__dict__.keys()

    def iloc(self, slices=None, **kwargs):
        result = copy(self)
        if slices is not None:
            kwargs.update(slices)
        for k, v in kwargs.items():
            result[k] = result[k].iloc[v]
        return result

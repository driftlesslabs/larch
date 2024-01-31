from __future__ import annotations


class SimpleAttribute:
    """
    A simple attribute descriptor.

    Use this class to create a simple attribute descriptor that circumvents
    a block on arbitrary attribute assignment.
    """

    def __init__(self, default: type = None):
        self.default = default

    def __get__(self, instance, owner):
        if instance is None:
            return self
        else:
            try:
                result = getattr(instance, self.private_name)
            except AttributeError:
                if self.default is None:
                    result = None
                else:
                    setattr(instance, self.private_name, self.default())
                    result = getattr(instance, self.private_name)
            return result

    def __set__(self, instance, values):
        if instance is not None:
            if self.default is not None and not isinstance(values, self.default):
                raise TypeError(
                    f"when assigning to {instance.__class__.__name__}.{self.name}, "
                    f"expected {self.default.__name__}, got {type(values).__name__}"
                )
            setattr(instance, self.private_name, values)

    def __delete__(self, instance):
        if instance is not None:
            delattr(instance, self.private_name)

    def __set_name__(self, owner, name):
        self.name = name
        self.private_name = "_" + name

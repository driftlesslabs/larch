from __future__ import annotations

from ._optional import jax


def reset_compiled_methods(obj):
    precompiled_funcs = [i for i in obj.__dict__ if i.startswith("_precompiled_")]
    for i in precompiled_funcs:
        delattr(obj, i)


class compiledmethod:
    """Decorator for a class method that returns a compiled function."""

    def __init__(self, compiler):
        # self : compiledmethod
        # compiler : the class method being decorated
        self.compiler = compiler
        self.docstring = compiler.__doc__

    def __set_name__(self, owner, name):
        # self : compiledmethod
        # owner : parent class that will have `self` as a member
        # name : the name of the attribute that `self` will be
        self.public_name = name
        self.private_name = "_precompiled_" + name

    def __get__(self, obj, objtype=None):
        # self : compiledmethod
        # obj : instance of parent class that has `self` as a member
        # objtype : class of `obj`
        unmangle = getattr(obj, "unmangle", None)
        if unmangle is not None:
            unmangle()
        result = getattr(obj, self.private_name, None)
        if result is None:
            result = self.compiler(obj)
            result.__doc__ = self.docstring
            setattr(obj, self.private_name, result)
        return result

    def __set__(self, obj, value):
        # self : compiledmethod
        # obj : instance of parent class that has `self` as a member
        # value : the new value that is trying to be assigned
        raise AttributeError(f"can't set {self.public_name}")

    def __delete__(self, obj):
        # self : compiledmethod
        # obj : instance of parent class that has `self` as a member
        setattr(obj, self.private_name, None)


class jitmethod:
    """Decorator for a class method that returns a compiled function."""

    def __init__(self, wrapped_method=None, **kwargs):
        """
        Initialize a jit compile wrapper.

        Parameters
        ----------
        wrapped_method : Callable
            The class method being decorated.
        """
        if wrapped_method is None:
            self.jit_kwargs = kwargs
        else:
            self.jit_kwargs = {}
            self.wrapped_method = wrapped_method
            self.docstring = wrapped_method.__doc__

    def __call__(self, wrapped_method):
        self.wrapped_method = wrapped_method
        self.docstring = wrapped_method.__doc__
        return self

    def __set_name__(self, owner, name):
        """
        Triggered on assignment to a class name.

        Parameters
        ----------
        owner : Any
            Parent class that will have `self` as a member.
        name : str
            The name of the attribute to which self is being assigned.
        """
        self.public_name = name
        self.private_name = "_precompiled_" + name

    def __get__(self, obj, objtype=None):
        """
        Access to jit'ed method.

        Parameters
        ----------
        obj : Any
            Instance of parent class that has `self` as a member.
        objtype : class
            Class of `obj`.
        """
        unmangle = getattr(obj, "unmangle", None)
        if unmangle is not None:
            unmangle()
        result = getattr(obj, self.private_name, None)
        if result is None:

            def func(*args, **kwargs):
                return self.wrapped_method(obj, *args, **kwargs)

            result = jax.jit(func, **self.jit_kwargs)
            result.__doc__ = self.docstring
            setattr(obj, self.private_name, result)
        return result

    def __set__(self, obj, value):
        raise AttributeError(f"can't set {self.public_name} is is a jitmethod")

    def __delete__(self, obj):
        """
        Clear precompiled jit'ed method.

        Parameters
        ----------
        obj : Any
            Instance of parent class that has `self` as a member.
        """
        setattr(obj, self.private_name, None)

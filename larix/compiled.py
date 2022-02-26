

def _reset_compiled_methods(obj):
    for name in obj._compiledmethods:
        delattr(obj, name)


class compiledmethod:
    """
    Decorator for a class method that returns a compiled function.
    """

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
        self.private_name = '_precompiled_' + name
        if not hasattr(owner, '_compiledmethods'):
            owner._compiledmethods = []
        owner._compiledmethods.append(name)

    def __get__(self, obj, objtype=None):
        # self : compiledmethod
        # obj : instance of parent class that has `self` as a member
        # objtype : class of `obj`
        unmangle = getattr(obj, 'unmangle', None)
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


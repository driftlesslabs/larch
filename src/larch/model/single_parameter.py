from __future__ import annotations


class SingleParameter:
    def __set_name__(self, owner, name):
        self.name = name
        self.private_name = "_" + name

    def __get__(self, instance, instancetype=None):
        """
        Get attribute from the instance.

        Parameters
        ----------
        instance : Any
            Instance of parent class that has `self` as a member.
        instancetype : class
            Class of `instance`.
        """
        if instance is None:
            return self
        return getattr(instance, self.private_name, None)

    def __set__(self, instance, value):
        """
        Get attribute on the instance.

        Parameters
        ----------
        instance : Any
            Instance of parent class that has `self` as a member.
        value : Any
            Value being set.
        """
        existing_value = self.__get__(instance)
        if value is not None:
            value = str(value)
        if existing_value != value:
            setattr(instance, self.private_name, value)
            try:
                instance.mangle()
            except AttributeError:
                pass

    def __delete__(self, instance):
        self.__set__(instance, None)

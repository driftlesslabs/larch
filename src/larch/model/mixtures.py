from __future__ import annotations

from collections.abc import MutableSequence

from .._optional import jax, jnp, js
from .param_core import ParameterBucket


class Mixture:
    def __init__(self):
        self._parent = None

    def param_names(self):
        """
        Named parameters referenced by this mixture, and their default values.

        Returns
        -------
        dict
        """
        raise NotImplementedError()

    def prep(self, bucket: ParameterBucket):
        raise NotImplementedError()

    def roll(self, draws: jax.Array, parameters: jax.Array) -> jax.Array:
        """
        Apply this mixing distribution to some random draws.

        Parameters
        ----------
        draws : jax.Array, shape [...]
            A set of pseudo-random draws, nominally uniformly distributed
            in the range 0 to 1.
        parameters : jax.Array, shape [..., n_params]
            An array of parameters, previously broadcasted to the same shape
            as the draws, plus the parameter dimension itself.

        Returns
        -------
        parameters : jax.Array, shape [..., n_params]
            The computed distribution of the target parameter has been overlaid.
        """
        raise NotImplementedError()

    def to_dict(self):
        raise NotImplementedError()


class MixtureList(MutableSequence):
    def __init__(self, init=None):
        self._parent = None
        self._mixtures = list()
        if init is not None:
            for i in init:
                if isinstance(i, Mixture):
                    self._mixtures.append(i)
                else:
                    raise TypeError(
                        f"members of {self.__class__.__name__} must be Mixture"
                    )

    def set_parent(self, instance):
        self._parent = instance
        for i in self._mixtures:
            i._parent = instance

    def __fresh(self, instance):
        newself = MixtureList()
        newself._instance = instance
        setattr(instance, self.private_name, newself)
        return newself

    def __mangle(self):
        try:
            self._parent.mangle()
        except AttributeError:
            pass

    def __get__(self, instance, owner):
        if instance is None:
            return self
        newself = getattr(instance, self.private_name, None)
        if newself is None:
            newself = self.__fresh(instance)
        return newself

    def __set__(self, instance, values):
        newself = getattr(instance, self.private_name, None)
        if newself is None:
            newself = self.__fresh(instance)
        else:
            newself._mixtures.clear()
        newself.__init__(values)
        newself.set_parent(instance)
        newself.__mangle()

    def __delete__(self, instance):
        newself = getattr(instance, self.private_name, None)
        if newself is not None and len(newself):
            newself.__mangle()
        if newself is None:
            newself = self.__fresh(instance)
        else:
            newself._mixtures.clear()
        newself.__init__()

    def __set_name__(self, owner, name):
        self.name = name
        self.private_name = "_private_" + name

    def __getitem__(self, item):
        return self._mixtures[item]

    def __setitem__(self, key: int, value):
        if not isinstance(value, Mixture):
            raise TypeError("items must be of type Mixture")
        self._mixtures[key] = value
        self.__mangle()
        self._mixtures[key]._parent = self._parent

    def __delitem__(self, key):
        del self._mixtures[key]
        self.__mangle()

    def __len__(self):
        return len(self._mixtures)

    def insert(self, index, value):
        if not isinstance(value, Mixture):
            raise TypeError("items must be of type Mixture")
        self._mixtures.insert(index, value)
        self.__mangle()
        self._mixtures[index]._parent = self._parent

    def __repr__(self):
        return repr(self._mixtures)

    def _is_duplicate(self, value):
        for i in self._mixtures:
            if i == value:
                return True
        return False

    def to_list(self):
        return [i.to_dict() for i in self._mixtures]

    def from_list(self, j):
        self._mixtures.clear()
        for i in j:
            kind = i.pop("type")
            if kind is None:
                raise ValueError("missing mixture type")
            cls = globals()[kind]
            self._mixtures.append(cls(**i))


class Normal(Mixture):
    def __init__(self, mean: str, std: str):
        super().__init__()
        self.mean_ = mean
        self.std_ = std
        self.imean = -1
        self.istd = -1
        self.default_mean = 0.0
        self.default_std = 0.001

    def __repr__(self):
        return f"{self.__class__.__name__}({self.mean_!r}, {self.std_!r})"

    def __eq__(self, other):
        return (
            isinstance(other, Normal)
            and self.mean_ == other.mean_
            and self.std_ == other.std_
        )

    def param_names(self):
        return {
            self.mean_: self.default_mean,
            self.std_: self.default_std,
        }

    def prep(self, bucket: ParameterBucket):
        self.imean = bucket.get_param_loc(self.mean_)
        self.istd = bucket.get_param_loc(self.std_)

    def roll(self, draw_vec, parameters):
        assert self.imean >= 0
        assert self.istd >= 0
        v = js.stats.norm.ppf(
            draw_vec, parameters[..., self.imean], parameters[..., self.istd]
        )
        parameters = parameters.at[..., self.imean].set(v)
        return parameters

    def to_dict(self):
        return dict(
            type=self.__class__.__name__,
            mean=self.mean_,
            std=self.std_,
        )


class LogNormal(Normal):
    def roll(self, draw_vec, parameters):
        assert self.imean >= 0
        assert self.istd >= 0
        v = js.stats.norm.ppf(
            draw_vec, parameters[..., self.imean], parameters[..., self.istd]
        )
        parameters = parameters.at[..., self.imean].set(jnp.exp(v))
        return parameters


class NegLogNormal(Normal):
    def roll(self, draw_vec, parameters):
        assert self.imean >= 0
        assert self.istd >= 0
        v = js.stats.norm.ppf(
            draw_vec, parameters[..., self.imean], parameters[..., self.istd]
        )
        parameters = parameters.at[..., self.imean].set(-jnp.exp(v))
        return parameters

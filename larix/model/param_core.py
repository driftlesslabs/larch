import logging

import numpy as np
import xarray as xr
from numbers import Number

def new_name(existing_names):
    n = 1
    while f"model{n:05d}" in existing_names:
        n += 1
    return f"model{n:05d}"


def empty(dtype, index_name):
    return xr.DataArray(np.empty(0, dtype=dtype), dims=index_name)


class ParameterBucket:

    index_name = "param_name"
    _data_types = {
        "value": np.float32,
        "initvalue": np.float32,
        "nullvalue": np.float32,
        "minimum": np.float32,
        "maximum": np.float32,
        "holdfast": np.int8,
    }

    def __init__(self, *models, **kmodels):
        self._models = {}
        self._params = xr.Dataset(
            {
                "value": empty(np.float32, self.index_name),
                "initvalue": empty(np.float32, self.index_name),
                "nullvalue": empty(np.float32, self.index_name),
                "minimum": empty(np.float32, self.index_name),
                "maximum": empty(np.float32, self.index_name),
                "holdfast": empty(np.int8, self.index_name),
            },
            coords={self.index_name: empty(np.object_, self.index_name)},
        )
        self._fill_values = {
            "value": 0.0,
            "initvalue": 0.0,
            "nullvalue": 0.0,
            "minimum": -np.inf,
            "maximum": np.inf,
            "holdfast": np.int8(0),
        }
        for k, m in kmodels.items():
            self.attach_model(m, k, agg=False)
        for m in models:
            if isinstance(m, dict):
                for k_, m_ in m.items():
                    self.attach_model(m_, k_, agg=False)
            else:
                self.attach_model(m, agg=False)
        self._aggregate_parameters()

    def attach_model(self, model, name=None, agg=True, unmangle=True):
        if name is None:
            name = new_name(self._models.keys())
        if unmangle:
            model.unmangle()
        self._models[name] = model
        model._name_in_parameter_bucket = name
        if agg:
            self._aggregate_parameters()

    def detach_model(self, model):
        name = getattr(model, '_name_in_parameter_bucket', None)
        if name is None:
            return
        elif self._models[name] is model:
            del self._models[name]

    def _aggregate_parameters(self):
        should_mangle = False
        all_names = set()
        for m in self._models.values():
            all_names |= set(m.pnames)
        new_params = self._params.reindex(
            {self.index_name: sorted(all_names)}, fill_value=self._fill_values
        )
        if new_params[self.index_name].size != self._params[
            self.index_name
        ].size or any(new_params[self.index_name] != self._params[self.index_name]):
            should_mangle = True
        self._params = new_params
        if should_mangle:
            self.mangle()

    def _preferred_data_type(self, key):
        t = self._data_types.get(key, None)
        if t is None:
            if key in self._params:
                t = self._params[key].dtype
        if t is None:
            t = np.object_
        return t

    def _clean_dtypes(self, new_params):
        for k in new_params:
            new_params = new_params.assign({k: new_params[k].astype(self._preferred_data_type(k))})
        return new_params

    def update_parameters(self, other):
        should_mangle = False
        joint_names = set(self.pnames) | set(other.pnames)
        new_params = self._params.reindex(
            {self.index_name: sorted(joint_names)}
        ).fillna(other)
        if new_params[self.index_name].size != self._params[
            self.index_name
        ].size or any(new_params[self.index_name] != self._params[self.index_name]):
            should_mangle = True
        new_params = self._clean_dtypes(new_params)
        self._params = new_params
        if should_mangle:
            self.mangle()

    def add_parameters(self, names, fill_values=None):
        should_mangle = False
        joint_names = set(self.pnames) | set(names)
        fills = self._fill_values.copy()
        if fill_values is not None:
            fills.update(fill_values)
        new_params = self._params.reindex(
            {self.index_name: sorted(joint_names)}, fill_value=fills
        )
        if new_params[self.index_name].size != self._params[
            self.index_name
        ].size or any(new_params[self.index_name] != self._params[self.index_name]):
            should_mangle = True
        new_params = self._clean_dtypes(new_params)
        self._params = new_params
        if should_mangle:
            self.mangle()

    def mangle(self):
        for m in self._models.values():
            m.mangle()
            m._ensure_names(self.pnames)

    def unmangle(self, *args, **kwargs):
        for m in self._models.values():
            m.unmangle(*args, **kwargs)

    @property
    def parameters(self):
        return self._params

    @property
    def n_params(self):
        return self._params.dims[self.index_name]

    @property
    def pvals(self):
        return self._params["value"].to_numpy()

    @pvals.setter
    def pvals(self, x):
        if isinstance(x, str):
            if x in self._params:
                x = self._params[x].to_numpy()
            elif f"{x}value" in self._params:
                x = self._params[f"{x}value"].to_numpy()
            elif f"{x}imum" in self._params:
                x = self._params[f"{x}imum"].to_numpy()
            else:
                raise ValueError(f"unknown value set {x}")
        if isinstance(x, dict):
            candidates = xr.DataArray(
                np.asarray(list(x.values())),
                dims=self.index_name,
                coords={self.index_name: np.asarray(list(x.keys()))}
            ).reindex({self.index_name: self.pnames}).fillna(self.pvals)
            x = np.where(
                self._params["holdfast"].to_numpy(),
                self._params["value"].to_numpy(),
                candidates.to_numpy(),
            )
        if isinstance(x, Number):
            candidates = xr.full_like(self._params["value"], x)
            x = np.where(
                self._params["holdfast"].to_numpy(),
                self._params["value"].to_numpy(),
                candidates.to_numpy(),
            )
        self._params = self._params.assign(
            {"value": xr.DataArray(x, dims=self._params["value"].dims)}
        )

    @property
    def pholdfast(self):
        return self._params["holdfast"].to_numpy()

    @pholdfast.setter
    def pholdfast(self, x):
        if isinstance(x, dict):
            x = xr.DataArray(
                np.asarray(list(x.values())),
                dims=self.index_name,
                coords={self.index_name: np.asarray(list(x.keys()))}
            ).reindex({self.index_name: self.pnames}).fillna(self.pholdfast).astype(np.int8)
        self._params = self._params.assign(
            {"holdfast": xr.DataArray(x, dims=self._params["holdfast"].dims)}
        )

    @property
    def pmaximum(self):
        return self._params["maximum"].to_numpy()

    @property
    def pminimum(self):
        return self._params["minimum"].to_numpy()

    @property
    def pnames(self):
        return self._params[self.index_name].to_numpy()

    def get_param_loc(self, name):
        """
        Get the position of a named parameter.

        Parameters
        ----------
        name : str
            Name of parameter to find

        Returns
        -------
        int
            Position of parameter in array

        Raises
        ------
        KeyError
            The parameter is not found
        """
        name = str(name)
        return self._params.indexes[self.index_name].get_loc(name)

    @property
    def pstderr(self):
        if "std_err" in self._params:
            return self._params["std_err"].to_numpy()
        else:
            return np.full_like(self.pvals, np.nan)

    @pstderr.setter
    def pstderr(self, x):
        self._params = self._params.assign(
            {"std_err": xr.DataArray(x, dims=self._params["value"].dims)}
        )

    def __getitem__(self, item):
        if item in self._models:
            return self._models[item]
        raise KeyError

    def _assign(self, key, param_name, value):
        self._params = self._params.assign({
            key: self._params[key].where(
                self._params[self.index_name] != param_name,
                value
            )
        })

    def lock(self, values=None, **kwargs):
        if values is None:
            values = {}
        values.update(kwargs)
        for k, v in values.items():
            self._assign("value", k, v)
            self._assign("initvalue", k, v)
            self._assign("holdfast", k, 1)

    def __set_name__(self, owner, name):
        self.name = name
        self.private_name = "_" + name

    def __get__(self, instance, owner=None):
        """
        Parameters
        ----------
        instance : Any
            Instance of parent class that has `self` as a member.
        instancetype : class
            Class of `instance`.
        """
        if instance is None:
            return self
        bucket = getattr(instance, self.private_name, None)
        return bucket

    def __set__(self, instance, value):
        if not isinstance(value, ParameterBucket):
            raise TypeError(f"{self.name} must be a ParameterBucket")
        existing = getattr(instance, self.private_name, None)
        if existing is value:
            return
        if existing is not None:
            existing.detach_model(instance)
        _name_in_parameter_bucket = getattr(instance, '_name_in_parameter_bucket', None)
        value.attach_model(instance, _name_in_parameter_bucket, agg=False, unmangle=False)
        if existing is not None:
            value.update_parameters(existing)
        setattr(instance, self.private_name, value)
        instance.mangle()

    def __delete__(self, instance):
        existing = getattr(instance, self.private_name, None)
        if existing is not None:
            existing.detach_model(instance)
        setattr(instance, self.private_name, None)
        instance.mangle()


from __future__ import annotations

import logging
from numbers import Number
from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from larch import BaseModel


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

    def attach_model(
        self,
        model: BaseModel,
        name: str | None = None,
        agg: bool = True,
        unmangle: bool | Literal["structure":] = True,
    ):
        """
        Attach a model to the bucket.

        Parameters
        ----------
        model : BaseModel
            The model to attach.
        name : str, optional
            The name to attach the model as.  If None, the model's ident
            attribute is used.  If the model has no ident attribute, a new
            name is generated.
        agg : bool
            If True, aggregate the parameters of the model with the bucket.
        unmangle : bool or str
            If True, unmangle the model.  If "structure", only unmangle the
            structure of the model.
        """
        if name is None:
            try:
                name = model.ident
            except AttributeError:
                pass
        else:
            self.ident = name
        if name is None:
            raise ValueError("missing name")
            # name = new_name(self._models.keys())
        if unmangle == "structure":
            model.unmangle(structure_only=True)
        elif unmangle:
            model.unmangle()
        self._models[name] = model

        # collect parameters from the incoming model to be attached, to be able
        # to update the bucket's parameters if they are not already present
        move_values = {}
        for k in model._parameter_bucket.pnames:
            if k not in self._params[self.index_name]:
                move_values[k] = model._parameter_bucket.parameters.sel(
                    {self.index_name: k}
                ).copy(deep=True)
        model._parameter_bucket = self
        if agg:
            if move_values:
                self.combine_parameters(
                    xr.concat(move_values.values(), dim="param_name")
                )
            self._aggregate_parameters()

    def detach_model(self, model):
        name = getattr(model, "_ident", None)
        # if getattr(self._models[name], "_parameter_bucket", None) is self:
        #     del self._models[name]._parameter_bucket
        if name is None:
            return
        elif self._models.get(name, None) is model:
            del self._models[name]
        if getattr(model, "_parameter_bucket", None) is self:
            del model._parameter_bucket

    def rename_model(self, oldname, newname):
        if oldname == newname:
            return
        try:
            m = self._models[oldname]
        except KeyError:
            print("keys:", list(self._models.keys()))
            raise
        if newname in self._models:
            raise ValueError("name exists")
        self._models[newname] = m
        del self._models[oldname]

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
            new_params = new_params.assign(
                {k: new_params[k].astype(self._preferred_data_type(k))}
            )
        return new_params

    def update_parameters(self, other):
        if isinstance(other, ParameterBucket):
            other = other._params
        if isinstance(other, xr.Dataset):
            if "param_name_2" in other.sizes:
                other = other.drop_dims("param_name_2")  # TODO, keep 2nd dim
            if len(other.sizes) != 1:
                raise ValueError("expected parameters to be a 1-d vector")
            other_dim = iter(other.sizes).__next__()
            joint_names = set(self.pnames) | set(other.coords[other_dim].data)
        else:
            joint_names = set(self.pnames) | set(other.pnames)
        should_mangle = False
        new_params = self._params.reindex({self.index_name: sorted(joint_names)})
        filling = [i for i in new_params if i in other]
        try:
            new_params = new_params.fillna(other[filling])
        except ValueError as err:
            logging.getLogger(__name__).error(f"other={other}")
            logging.getLogger(__name__).exception(f"err={err}")
        if new_params[self.index_name].size != self._params[
            self.index_name
        ].size or any(new_params[self.index_name] != self._params[self.index_name]):
            should_mangle = True
        new_params = self._clean_dtypes(new_params)
        self._params = new_params
        if should_mangle:
            self.mangle()

    def combine_parameters(self, other: xr.Dataset | ParameterBucket):
        """
        Combine the parameters with another set of parameters.

        Values in the current parameter set are replaced by values in the
        other parameter set.
        """
        if isinstance(other, ParameterBucket):
            other = other._params
        if isinstance(other, xr.Dataset):
            if "param_name_2" in other.sizes:
                other = other.drop_dims("param_name_2")
        if len(other.sizes) != 1:
            raise ValueError("expected parameters to be a 1-d vector")
        if not isinstance(other, xr.Dataset):
            raise TypeError("expected other to be a Dataset")
        should_mangle = False
        n_existing_params = self._params.sizes[self.index_name]
        self._params = other.combine_first(self._params)
        if self._params.sizes[self.index_name] != n_existing_params:
            should_mangle = True
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

    def add_array(self, name, values):
        for j in range(values.ndim):
            if values.shape[j] != self.n_params:
                raise ValueError(f"cannot add array, unexpected shape {values.shape}")
        if values.ndim == 1:
            dims = [self.index_name]
        else:
            dims = [self.index_name] + [
                f"{self.index_name}_{j + 2}" for j in range(values.ndim - 1)
            ]
        self._params = self._params.assign(
            {name: xr.DataArray(np.asarray(values), dims=dims)}
        )

    def add_parameter_array(self, name, values):
        return self.add_array(name, values)

    def mangle(self, data=True, structure=True):
        for m in self._models.values():
            m.mangle(data=data, structure=structure)
            m._ensure_names(self.pnames)

    def unmangle(self, *args, **kwargs):
        for m in self._models.values():
            m.unmangle(*args, **kwargs)

    def _scan_all_ensure_names(self):
        for m in self._models.values():
            m._scan_all_ensure_names()

    @property
    def parameters(self):
        return self._params

    @property
    def n_params(self):
        return self._params.sizes[self.index_name]

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
            candidates = (
                xr.DataArray(
                    np.asarray(list(x.values())),
                    dims=self.index_name,
                    coords={self.index_name: np.asarray(list(x.keys()))},
                )
                .reindex({self.index_name: self.pnames})
                .fillna(self.pvals)
            )
            x = np.where(
                self._params["holdfast"].to_numpy(),
                self._params["value"].to_numpy(),
                candidates.to_numpy(),
            )
        elif isinstance(x, Number):
            candidates = xr.full_like(self._params["value"], x)
            x = np.where(
                self._params["holdfast"].to_numpy(),
                self._params["value"].to_numpy(),
                candidates.to_numpy(),
            )
        else:
            # we are getting an array
            x = np.asanyarray(x).reshape(-1)
            assert x.size == self._params["value"].size
            # do not allow changing holdfast values
            x = np.where(
                self._params["holdfast"].to_numpy(),
                self._params["value"].to_numpy(),
                x,
            )
        self._params = self._params.assign(
            {"value": xr.DataArray(x, dims=self._params["value"].dims)}
        )

    @property
    def pinitvals(self):
        return self._params["initvalue"].to_numpy()

    @pinitvals.setter
    def pinitvals(self, x):
        if isinstance(x, str):
            raise ValueError("cannot set initvals with a string")
        if isinstance(x, dict):
            candidates = (
                xr.DataArray(
                    np.asarray(list(x.values())),
                    dims=self.index_name,
                    coords={self.index_name: np.asarray(list(x.keys()))},
                )
                .reindex({self.index_name: self.pnames})
                .fillna(self.pvals)
            )
            x = np.where(
                self._params["holdfast"].to_numpy(),
                self._params["initvalue"].to_numpy(),
                candidates.to_numpy(),
            )
        elif isinstance(x, Number):
            candidates = xr.full_like(self._params["value"], x)
            x = np.where(
                self._params["holdfast"].to_numpy(),
                self._params["initvalue"].to_numpy(),
                candidates.to_numpy(),
            )
        else:
            # we are getting an array
            x = np.asanyarray(x).reshape(-1)
            assert x.size == self._params["initvalue"].size
            # do not allow changing holdfast values
            x = np.where(
                self._params["holdfast"].to_numpy(),
                self._params["initvalue"].to_numpy(),
                x,
            )
        self._params = self._params.assign(
            {"initvalue": xr.DataArray(x, dims=self._params["initvalue"].dims)}
        )

    @property
    def pholdfast(self):
        return self._params["holdfast"].to_numpy()

    @property
    def pnullvals(self):
        return self._params["nullvalue"].to_numpy()

    @pholdfast.setter
    def pholdfast(self, x):
        if isinstance(x, dict):
            x = (
                xr.DataArray(
                    np.asarray(list(x.values())),
                    dims=self.index_name,
                    coords={self.index_name: np.asarray(list(x.keys()))},
                )
                .reindex({self.index_name: self.pnames})
                .fillna(self.pholdfast)
                .astype(np.int8)
            )
        self._params = self._params.assign(
            {"holdfast": xr.DataArray(x, dims=self._params["holdfast"].dims)}
        )

    @pnullvals.setter
    def pnullvals(self, x):
        if isinstance(x, dict):
            x = (
                xr.DataArray(
                    np.asarray(list(x.values())),
                    dims=self.index_name,
                    coords={self.index_name: np.asarray(list(x.keys()))},
                )
                .reindex({self.index_name: self.pnames})
                .fillna(self.pnullvals)
            )
        self._params = self._params.assign(
            {"nullvalue": xr.DataArray(x, dims=self._params["nullvalue"].dims)}
        )

    @property
    def pmaximum(self):
        return self._params["maximum"].to_numpy()

    @pmaximum.setter
    def pmaximum(self, x):
        if isinstance(x, dict):
            candidates = (
                xr.DataArray(
                    np.asarray(list(x.values())),
                    dims=self.index_name,
                    coords={self.index_name: np.asarray(list(x.keys()))},
                )
                .reindex({self.index_name: self.pnames})
                .fillna(self.pmaximum)
            )
            x = np.where(
                self._params["holdfast"].to_numpy(),
                self._params["maximum"].to_numpy(),
                candidates.to_numpy(),
            )
        if isinstance(x, Number):
            candidates = xr.full_like(self._params["value"], x)
            x = np.where(
                self._params["holdfast"].to_numpy(),
                self._params["maximum"].to_numpy(),
                candidates.to_numpy(),
            )
        self._params = self._params.assign(
            {"maximum": xr.DataArray(x, dims=self._params["maximum"].dims)}
        )

    @property
    def pminimum(self):
        return self._params["minimum"].to_numpy()

    @pminimum.setter
    def pminimum(self, x):
        if isinstance(x, dict):
            candidates = (
                xr.DataArray(
                    np.asarray(list(x.values())),
                    dims=self.index_name,
                    coords={self.index_name: np.asarray(list(x.keys()))},
                )
                .reindex({self.index_name: self.pnames})
                .fillna(self.pminimum)
            )
            x = np.where(
                self._params["holdfast"].to_numpy(),
                self._params["minimum"].to_numpy(),
                candidates.to_numpy(),
            )
        if isinstance(x, Number):
            candidates = xr.full_like(self._params["value"], x)
            x = np.where(
                self._params["holdfast"].to_numpy(),
                self._params["minimum"].to_numpy(),
                candidates.to_numpy(),
            )
        self._params = self._params.assign(
            {"minimum": xr.DataArray(x, dims=self._params["minimum"].dims)}
        )

    @property
    def pbounds(self):
        """scipy.optimize.Bounds : A copy of the current min-max bounds of the parameters."""
        self.unmangle()
        from scipy.optimize import Bounds

        return Bounds(
            self.pminimum,
            self.pmaximum,
        )

    def set_cap(self, cap=25):
        """
        Set limiting values for one or more parameters.

        Parameters
        ----------
        cap : numeric, default 25.0
            Set a global limit on parameters.  The maximum has a ceiling
            at this value, and the minimum a floor at the negative of this, unless
            the existing bounds are entirely outside this range.
        """
        # propose capping all parameters
        minimum_ = np.maximum(self.pminimum, -cap)
        maximum_ = np.minimum(self.pmaximum, cap)

        # anywhere the minmax range was fully collapsed, restore the originals
        minimum_ = np.where(minimum_ <= maximum_, minimum_, self.pminimum)
        maximum_ = np.where(minimum_ <= maximum_, maximum_, self.pmaximum)

        # set any NaN bounds to the cap
        minimum_ = np.where(np.isnan(minimum_), -cap, minimum_)
        maximum_ = np.where(np.isnan(maximum_), cap, maximum_)

        # restore the originals for holdfast parameters
        minimum_ = np.where(self.pholdfast, self.pminimum, minimum_)
        maximum_ = np.where(self.pholdfast, self.pmaximum, maximum_)

        # write out new bounds
        self._params = self._params.assign(
            {
                "minimum": xr.DataArray(minimum_, dims=self._params["value"].dims),
                "maximum": xr.DataArray(maximum_, dims=self._params["value"].dims),
            }
        )

    @property
    def pnames(self):
        return self._params[self.index_name].to_numpy()

    def get_param_loc(self, name) -> int:
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
        self._params = self._params.assign(
            {
                key: self._params[key].where(
                    self._params[self.index_name] != param_name, value
                )
            }
        )

    def lock(self, values=None, **kwargs) -> None:
        """
        Lock the values of one or more parameters.

        Parameters
        ----------
        values : dict, optional
            Dictionary of parameters to lock.  The keys are parameter names
            and the values are the values to set as locked.
        kwargs : dict
            Additional parameters to lock.
        """
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
        bucket = getattr(instance, self.private_name, None)
        return bucket

    def __set__(self, instance, value):
        """
        Set attribute on the instance.

        Parameters
        ----------
        instance : Any
            Instance of parent class that has `self` as a member.
        value : ParameterBucket
            The bucket being assigned.
        """
        if not isinstance(value, ParameterBucket):
            raise TypeError(f"{self.name} must be a ParameterBucket")
        existing = getattr(instance, self.private_name, None)
        if existing is value:
            return
        if existing is not None:
            existing.detach_model(instance)
        # _name_in_parameter_bucket = getattr(instance, "_name_in_parameter_bucket", None)
        setattr(instance, self.private_name, value)
        instance.mangle()
        value.attach_model(instance, agg=False, unmangle="structure")
        if existing is not None:
            value.update_parameters(existing)
        instance.mangle()

    def __delete__(self, instance):
        existing = getattr(instance, self.private_name, None)
        setattr(instance, self.private_name, None)
        if existing is not None:
            existing.detach_model(instance)
        instance.mangle()

    def pretty_table(self, name_width=25):
        from rich.table import Table

        params = self._params
        rich_table = Table()
        idx_name, n = next(iter(params.sizes.items()))
        for column in params.variables:
            rich_table.add_column(
                str(column), width=name_width if column == self.index_name else None
            )
        for i in range(n):
            row = [str(params[j].data[i]) for j in params.variables]
            rich_table.add_row(*row)
        return rich_table

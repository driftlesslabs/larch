"""
Base model class and related functions for a discrete choice modeling framework.

The main class in this module is `BaseModel`, which serves as the base class for
all models in the framework. It contains a `ParameterBucket` for storing parameters,
a `DictOfLinearFunction` for computing utility from idco data, and a placeholder
for the model subtype.

The `BaseModel` class is designed to be subclassed by other model classes that
implement specific types of discrete choice models.
"""

from __future__ import annotations

import base64
import logging
import pathlib
import uuid
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from .._optional import jax
from ..dataset import DataTree
from ..exceptions import MissingDataError
from ..util.simple_attribute import SimpleAttribute
from .constraints import ParametricConstraintList
from .linear import DictOfAlts, DictOfLinearFunction, LinearFunction
from .mixtures import MixtureList
from .param_core import ParameterBucket
from .single_parameter import SingleParameter
from .tree import NestingTree

if TYPE_CHECKING:
    from scipy.optimize import Bounds
    from xarray import Dataset

logger = logging.getLogger("larch.model")

MANGLE_DATA = 0x1
MANGLE_STRUCTURE = 0x2


def _unique_ident():
    j = base64.b32encode(uuid.uuid4().bytes)[:25].decode()
    return f"{j[:5]}-{j[5:10]}-{j[10:15]}-{j[15:20]}-{j[20:]}"


class _ModelData:
    __slots__ = ("model",)

    def __set_name__(self, owner, name):
        assert name == "data"

    def __init__(self, instance: BaseModel | None = None):
        self.model = instance

    def get_dataref(self):
        """Dataset | DataTree : A source for data for the model.

        If the datatree is a single-node tree (i.e. a single Dataset), this
        property returns the root dataset.  Otherwise, it returns the full
        datatree.
        """
        if self.model.datatree is None:
            raise MissingDataError("no data available for model")
        if len(self.model.datatree.subspaces) == 1:
            return self.model.datatree.root_dataset
        else:
            return self.model.datatree

    def __getitem__(self, key):
        return self.get_dataref()[key]

    def __setitem__(self, key, value):
        self.get_dataref()[key] = value
        self.model.mangle(data=True, structure=False)

    def __delitem__(self, key):
        del self.get_dataref()[key]
        self.model.mangle(data=True, structure=False)

    def __getattr__(self, item):
        return getattr(self.get_dataref(), item)

    def __get__(self, instance: BaseModel, owner):
        if instance is None:
            return self
        return _ModelData(instance)

    def __set__(self, instance: BaseModel, value):
        if instance is None:
            raise AttributeError("can't set attribute")
        raise instance.swap_datatree(value, True)


class BaseModel:
    """Base class for discrete choice models."""

    _parameter_bucket = ParameterBucket()
    _model_subtype = None

    utility_co = DictOfLinearFunction()
    """DictOfLinearFunction : The portion of the utility function computed from |idco| data.

    The keys of this mapping are alternative codes for the applicable elemental
    alternatives, and the values are linear functions to compute for the indicated
    alternative.  Each alternative that has any idco utility components must have
    a unique linear function given.

    Examples
    --------

    >>> from larch import Model, P, X
    >>> m = Model()
    >>> m.utility_co = {
    ... 	1: P.ParamA1 * X.DataA,
    ... 	2: P.ParamA2 * X.DataA + P.ParamASC2,
    ... }
    >>> print(m.utility_co)
    DictOfLinearFunction({1: P.ParamA1 * X.DataA, 2: P.ParamA2 * X.DataA + P.ParamASC2})
    """

    utility_ca = LinearFunction()
    """LinearFunction : The portion of the utility function computed from |idca| data.

    Data expressions in this utility function can actually reference both |idca|
    and |idco| format variables. Except in unusual model designs, every complete data
    expression should have at least one |idca| component.

    Examples
    --------

    >>> from larch import Model, P, X
    >>> m = Model()
    >>> m.utility_ca = P.Param1 * X.Data1 + P.Param2 * X.Data2
    >>> print(m.utility_ca)
    P.Param1 * X.Data1 + P.Param2 * X.Data2
    >>> m.utility_ca += P.Param3 * X.Data3 / X.DataCO4
    >>> print(m.utility_ca)
    P.Param1 * X.Data1 + P.Param2 * X.Data2 + P.Param3 * X('Data3/DataCO4')
    """

    quantity_ca = LinearFunction()
    """LinearFunction : The portion of the quantity function computed from |idca| data.

    Data expressions in this utility function can actually reference both |idca|
    and |idco| format variables. Except in unusual model designs, every complete data
    expression should have at least one |idca| component.

    Note that for the quantity function, the actual computed linear function
    uses the exponential of the parameter value(s), not the raw values. Thus,
    if the quantity function is given as `P.Param1 * X.Data1 + P.Param2 * X.Data2`,
    the computed values will actually be `exp(P.Param1) * X.Data1 + exp(P.Param2) * X.Data2`.
    This transformation ensures that the outcome from the quantity function is
    always positive, so long as at all of the data terms in the function are
    positive.  The `LinearFunction` class itself is not intrinsically aware
    of this implementation detail, but the `Model.utility_functions()` method is,
    and will render the complete utility function in a mathematically correct form.

    Examples
    --------

    >>> from larch import Model, P, X
    >>> m = Model()
    >>> m.quantity_ca = P.Param1 * X.Data1 + P.Param2 * X.Data2
    >>> print(m.quantity_ca)
    P.Param1 * X.Data1 + P.Param2 * X.Data2
    >>> m.quantity_ca += P.Param3 * X.Data3 / X.DataCO4
    >>> print(m.quantity_ca)
    P.Param1 * X.Data1 + P.Param2 * X.Data2 + P.Param3 * X('Data3/DataCO4')
    """

    _graph = NestingTree()

    constraints = ParametricConstraintList()

    _availability_co_vars = DictOfAlts()
    _choice_co_vars = DictOfAlts()

    mixtures = MixtureList()

    title = SimpleAttribute()
    rename_parameters = SimpleAttribute(dict)
    ordering = SimpleAttribute()

    def __init__(
        self,
        *,
        title=None,
        datatree=None,
        compute_engine=None,
        submodels=None,
        named_submodels=None,
        use_streaming=False,
        cache_dir=None,
        autoscale_weights=False,
        graph: NestingTree | None = None,
        dataservice=NotImplemented,
    ):
        if dataservice is not NotImplemented:
            raise NotImplementedError(
                "dataservice is no longer supported, use `datatree` instead"
            )
        if not hasattr(self, "_ident"):
            self._ident = _unique_ident()
        self._mangled = 0x3
        self._datatree = None
        self.title = title
        self._compute_engine = compute_engine
        self._use_streaming = use_streaming
        self._autoscale_weights = autoscale_weights
        self._possible_overspecification = None

        if submodels is None:
            submodels = {}
        for k, m in submodels.items():
            m.ident = k
        if named_submodels is None:
            named_submodels = {}
        for k, m in named_submodels.items():
            m.ident = k
        bucket = ParameterBucket(submodels, **named_submodels)
        self._parameter_bucket = bucket
        self.datatree = datatree
        if cache_dir is not None:
            cache_dir = pathlib.Path(cache_dir)
            cache_dir.mkdir(exist_ok=True)
            self.datatree.cache_dir = cache_dir
        self._cached_loglike_best = None
        self._cached_loglike_null = None
        self._most_recent_estimation_result = None

        self._choice_ca_var = None
        self._choice_co_code = None
        self._choice_co_vars = None
        self._choice_any = None

        self._weight_co_var = None

        self._availability_ca_var = None
        self._availability_co_vars = None
        self._availability_any = True

        self.graph = graph

    def __setattr__(self, name, value):
        if name.startswith("_") or hasattr(self, "_" + name) or hasattr(self, name):
            object.__setattr__(self, name, value)
        else:
            raise TypeError(
                f"Cannot set {name!r} on object of type {self.__class__.__name__}"
            )

    @property
    def ident(self):
        """Getter method for the ident property."""
        return self._ident

    @ident.setter
    def ident(self, x):
        try:
            old_ident = self._ident
        except AttributeError:
            pass
        else:
            self._parameter_bucket.rename_model(old_ident, x)
        self._ident = x

    @property
    def compute_engine(self):
        return self._compute_engine

    @compute_engine.setter
    def compute_engine(self, engine):
        if engine not in {"numba", "jax", None}:
            raise ValueError("invalid compute engine")
        self._compute_engine = engine
        if self._compute_engine == "jax" and not jax:
            warnings.warn(
                "jax is not installed, falling back to numba",
                stacklevel=2,
            )
            self._compute_engine = "numba"
        if self._compute_engine == "jax" and self.use_streaming:
            warnings.warn(
                "setting use_streaming to False, jax is not yet compatible",
                stacklevel=2,
            )
            self.use_streaming = False

    @property
    def dataset(self) -> Dataset | None:
        """xarray.Dataset : Data arrays as loaded for model computation."""
        raise NotImplementedError("the BaseModel class does not include a dataset")

    @property
    def use_streaming(self):
        return self._use_streaming

    @use_streaming.setter
    def use_streaming(self, should):
        should = bool(should)
        if should and self.compute_engine != "numba":
            raise ValueError(
                "streaming is currently only compatible with the numba compute engine"
            )
        self._use_streaming = should

    @property
    def autoscale_weights(self) -> bool:
        """Whether to automatically scale case weights."""
        return self._autoscale_weights

    @autoscale_weights.setter
    def autoscale_weights(self, should: bool) -> None:
        temp = bool(should)
        if temp != self._autoscale_weights:
            self.mangle(data=True, structure=False)
        self._autoscale_weights = temp

    @property
    def most_recent_estimation_result(self):
        """A copy of the result dict from most recent likelihood maximization."""
        return self._most_recent_estimation_result.copy()

    @property
    def possible_overspecification(self):
        """Possible overspecification of the model."""
        if self._possible_overspecification is None:
            return None
        from ..util.overspec_viewer import OverspecView

        return OverspecView(self._possible_overspecification)

    @property
    def datatree(self):
        """DataTree : A source for data for the model."""
        return self._datatree

    @datatree.setter
    def datatree(self, tree):
        self.swap_datatree(tree, True)

    def swap_datatree(self, tree: DataTree | xr.Dataset, should_mangle=False) -> None:
        """
        Swap the current datatree with a new datatree.

        Parameters
        ----------
        tree : DataTree or xr.Dataset or None
            The new datatree to be swapped. If None, the datatree will be set to None.
        should_mangle : bool, optional
            A boolean indicating whether to mangle the structure after swapping the datatree.
            Default is False.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If the `tree` parameter is not of type DataTree or xr.Dataset.

        Notes
        -----
        If `tree` is a xarray.Dataset, it will be converted to a DataTree using
        the `as_tree` method of the Dataset.

        If `should_mangle` is True, the structure of the datatree will be mangled
        after swapping.

        Examples
        --------
        >>> model = BaseModel()
        >>> tree = DataTree()
        >>> model.swap_datatree(tree, should_mangle=True)
        """
        from ..dataset import DataTree

        if (
            tree is not None
            and hasattr(tree, "relationships_are_digitized")
            and (not tree.relationships_are_digitized)
            and self.use_streaming
        ):
            tree = tree.digitize_relationships()
        if tree is self.datatree:
            return
        if isinstance(tree, DataTree) or tree is None:
            self._datatree = tree
            if should_mangle:
                self.mangle(structure=False)
        elif isinstance(tree, xr.Dataset):
            self._datatree = tree.dc.as_tree()
            if should_mangle:
                self.mangle(structure=False)
        else:
            try:
                self._datatree = DataTree(main=xr.Dataset.construct(tree))
            except Exception as err:
                raise TypeError(f"datatree must be DataTree not {type(tree)}") from err
            else:
                if should_mangle:
                    self.mangle(structure=False)

    def load_data(self, *args, **kwargs) -> None:
        """
        No-op.

        This method is deprecated since version 6, as data is automatically
        prepared as needed.  Use the `datatree` property to set the data source.
        """
        warnings.warn(
            "Model.load_data() is deprecated, this method is no longer needed",
            DeprecationWarning,
            stacklevel=2,
        )

    data = _ModelData()

    def doctor(self, **kwargs):
        """
        Run diagnostics, checking for common problems and inconsistencies.

        See :func:`larch.model.troubleshooting.doctor` for more information.
        """
        self.unmangle()
        from .troubleshooting import doctor

        if len(kwargs) == 0:
            kwargs = {
                "repair_ch_av": "!",
                "repair_ch_zq": "!",
                "repair_av_zq": "!",
                "repair_noch_nzwt": "!",
                "repair_nan_wt": "!",
                "repair_nan_data_co": "!",
                "repair_nan_utility": "!",
                "check_low_variance_data_co": "?",
                "check_overspec": "?",
                "verbose": 3,
            }

        result = doctor(self, **kwargs)
        return result

    @property
    def parameters(self):
        self.unmangle()
        return self._parameter_bucket.parameters

    def update_parameters(self, x):
        self.unmangle()
        return self._parameter_bucket.update_parameters(x)

    def add_parameter_array(self, name, values):
        return self._parameter_bucket.add_array(name, values)

    @property
    def n_params(self):
        self.unmangle()
        return self._parameter_bucket.n_params

    @property
    def pvals(self) -> np.ndarray[np.float64]:
        """An array of the current parameter values.

        This property is a getter/setter for the current parameter values.  The
        array is a copy of the current parameter values, and setting the array
        with a given array (which must be a vector with length equal to the
        number of parameters in the model) will update the parameter values in
        the model.

        Values can also be set here using a dictionary of parameter names and
        values. A warning will be given if any key of the dictionary is not
        found among the existing named parameters in the parameter frame, and
        the value associated with that key is ignored.  Any parameters not
        named by a key in this dictionary are not changed.

        Setting this property also permits the use of shorthand (string) values,
        including "null", "init", and "best" to set the parameter values to the
        null, initial, or best values, respectively, as given in the parameter
        frame.  If the parameter frame does not have these columns, a ValueError
        exception will be raised.
        """
        self.unmangle()
        return self._parameter_bucket.pvals

    @pvals.setter
    def pvals(self, x: np.ndarray[np.float64] | dict[str, float] | str):
        self.unmangle()
        self._parameter_bucket.pvals = x

    @property
    def pinitvals(self) -> np.ndarray[np.float64]:
        """An array of the initial parameter values.

        This property is a getter/setter for the initial parameter values.  The
        array is a copy of the initial parameter values, and setting the array
        with a given array (which must be a vector with length equal to the
        number of parameters in the model) will update the initial values in
        the model.

        Values can also be set here using a dictionary of parameter names and
        initial values. A warning will be given if any key of the dictionary is not
        found among the existing named parameters in the parameter frame, and
        the value associated with that key is ignored.  Any parameters not
        named by a key in this dictionary are not changed.
        """
        self.unmangle()
        return self._parameter_bucket.pinitvals

    @pinitvals.setter
    def pinitvals(self, x: np.ndarray[np.float64] | dict[str, float]):
        self.unmangle()
        self._parameter_bucket.pinitvals = x

    @property
    def pnames(self) -> np.ndarray[np.str_]:
        """An array of the current parameter names."""
        self.unmangle()
        return self._parameter_bucket.pnames

    @property
    def pholdfast(self) -> np.ndarray[np.int8]:
        """An array indicating which parameters are marked as holdfast.

        Parameters marked as `holdfast` are not changed during estimation by the
        likelihood maximization algorithm, but they can be changed by the user.
        """
        self.unmangle()
        return self._parameter_bucket.pholdfast

    @pholdfast.setter
    def pholdfast(self, x):
        self.unmangle()
        self._parameter_bucket.pholdfast = x

    @property
    def pnullvals(self) -> np.ndarray[np.float64]:
        """An array of the current parameter null values."""
        self.unmangle()
        return self._parameter_bucket.pnullvals

    @pnullvals.setter
    def pnullvals(self, x):
        self.unmangle()
        self._parameter_bucket.pnullvals = x

    @property
    def pmaximum(self) -> np.ndarray[np.float64]:
        """An array of the current parameter maximum values.

        The maximum values are used to set the upper bounds of the parameters
        during estimation by the likelihood maximization algorithm.
        """
        self.unmangle()
        return self._parameter_bucket.pmaximum

    @pmaximum.setter
    def pmaximum(self, x):
        self.unmangle()
        self._parameter_bucket.pmaximum = x

    @property
    def pminimum(self) -> np.ndarray[np.float64]:
        """An array of the current parameter minimum values.

        The minimum values are used to set the lower bounds of the parameters
        during estimation by the likelihood maximization algorithm.
        """
        self.unmangle()
        return self._parameter_bucket.pminimum

    @pminimum.setter
    def pminimum(self, x):
        self.unmangle()
        self._parameter_bucket.pminimum = x

    @property
    def pbounds(self) -> Bounds:
        """scipy.optimize.Bounds : A copy of the current min-max bounds of the parameters."""
        self.unmangle()
        from scipy.optimize import Bounds

        return Bounds(
            self._parameter_bucket.pminimum,
            self._parameter_bucket.pmaximum,
        )

    @property
    def pstderr(self) -> np.ndarray[np.float64]:
        """An array of the current parameter standard errors."""
        self.unmangle()
        return self._parameter_bucket.pstderr

    @pstderr.setter
    def pstderr(self, x):
        self._parameter_bucket.pstderr = x

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
        self.unmangle()
        return self._parameter_bucket.set_cap(cap=cap)

    def plock(self, values=None, **kwargs) -> None:
        """
        Lock the values of one or more parameters.

        Parameters
        ----------
        values : dict, optional
            A dictionary of parameter names and values to lock.
        kwargs : dict
            The parameters to lock can alternatively be given as keyword arguments.
        """
        self.unmangle()
        self._parameter_bucket.lock(values, **kwargs)

    def get_param_loc(self, name) -> int:
        """
        Get the position of a named parameter.

        Parameters in a model are stored in an array, and this method returns
        the position of a named parameter in that array.

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
        self.unmangle()
        return self._parameter_bucket.get_param_loc(name)

    def get_value(self, name, *, default=None, kind="value"):
        if name is None and default is not None:
            return default
        if isinstance(name, dict):
            return {k: self.get_value(v) for k, v in name.items()}
        try:
            from .linear import LinearComponent, ParameterRef
            from .linear_math import ParameterOp

            if isinstance(name, (ParameterRef | ParameterOp)):
                return name.value(self)
            if isinstance(name, (LinearComponent | LinearFunction)):
                return name.as_pmath().value(self)
            if kind == "value":
                return self.pvals[self.get_param_loc(name)]
            return self.parameters[kind].values[self.get_param_loc(name)]
        except KeyError:
            if default is not None:
                return default
            else:
                raise

    @property
    def pf(self) -> pd.DataFrame:
        """
        pd.DataFrame: A DataFrame of the model parameters.

        The DataFrame has (potentially) columns for the parameter values, the
        best values, the initial values, the minimum and maximum values, and the
        null values.  If any of these are not present (i.e. because they have
        not been created yet), they are not included.
        """
        self.unmangle(structure_only=True)
        cols = [
            "value",
            "best",
            "initvalue",
            "minimum",
            "maximum",
            "nullvalue",
            "holdfast",
        ]
        cols = [i for i in cols if i in self._parameter_bucket._params]
        return self._parameter_bucket._params[cols].to_dataframe()

    def set_value(
        self,
        name,
        value: float | None = None,
        *,
        initvalue: float | None = None,
        nullvalue: float | None = None,
        minimum: float | None = None,
        maximum: float | None = None,
        holdfast: int | None = None,
    ):
        """
        Set the value one or more attributes of a single parameter.

        Parameters
        ----------
        name : str
            The name of the parameter to set.
        value : float, optional
            The value to set for the parameter. If not given, the value is not
            changed.
        initvalue : float, optional
            The initial value to set for the parameter. If not given, the
            initial value is not changed.
        nullvalue : float, optional
            The null value to set for the parameter. If not given, the value is
            not changed.
        minimum : float, optional
            The minimum value to set for the parameter. If not given, the value
            is not changed.
        maximum : float, optional
            The maximum value to set for the parameter. If not given, the value
            is not changed.
        holdfast : int, optional
            The holdfast value to set for the parameter. If not given, the value
            is not changed.
        """
        self.unmangle()
        if value is not None:
            self._parameter_bucket._assign("value", name, value)
        if initvalue is not None:
            self._parameter_bucket._assign("initvalue", name, initvalue)
        if nullvalue is not None:
            self._parameter_bucket._assign("nullvalue", name, nullvalue)
        if minimum is not None:
            self._parameter_bucket._assign("minimum", name, minimum)
        if maximum is not None:
            self._parameter_bucket._assign("maximum", name, maximum)
        if holdfast is not None:
            self._parameter_bucket._assign("holdfast", name, holdfast)

    def set_values(self, values=None, **kwargs):
        """
        Set the parameter values for one or more parameters.

        This method is deprecated since version 6.
        Use the `pvals` property instead.

        Parameters
        ----------
        values : {'null', 'init', 'best', array-like, dict, scalar}, optional
            New values to set for the parameters.
            If 'null' or 'init', the current values are set
            equal to the null or initial values given in
            the 'nullvalue' or 'initvalue' column of the
            parameter frame, respectively.
            If 'best', the current values are set equal to
            the values given in the 'best' column of the
            parameter frame, if that columns exists,
            otherwise a ValueError exception is raised.
            If given as array-like, the array must be a
            vector with length equal to the length of the
            parameter frame, and the given vector will replace
            the current values.  If given as a dictionary,
            the dictionary is used to update `kwargs` before
            they are processed.
        kwargs : dict
            Any keyword arguments (or if `values` is a
            dictionary) are used to update the included named
            parameters only.  A warning will be given if any key of
            the dictionary is not found among the existing named
            parameters in the parameter frame, and the value
            associated with that key is ignored.  Any parameters
            not named by key in this dictionary are not changed.

        Notes
        -----
        Setting parameters both in the `values` argument and
        through keyword assignment is not explicitly disallowed,
        although it is not recommended.

        """
        warnings.warn(
            "Model.set_values(x) is deprecated, use Model.pvals = x",
            DeprecationWarning,
            stacklevel=2,
        )
        if isinstance(values, dict):
            kwargs.update(values)
        elif values is None:
            self.pvals = kwargs
        else:
            self.pvals = values

    def lock_value(self, name=None, value=None, **kwargs):
        """
        Set a fixed value for a model parameter.

        Parameters with a fixed value (i.e., with "holdfast" set to 1)
        will not be changed during estimation by the likelihood
        maximization algorithm.

        Parameters
        ----------
        name : str
            The name of the parameter to set to a fixed value.
        value : float
            The numerical value to set for the parameter.
        **kwargs
            Alternatively, use (name=value, ...) form for multiple
            parameters.
        """
        from .linear import ParameterRef

        if name is None or value is None:
            if name is not None or value is not None or len(kwargs) == 0:
                raise ValueError("give name and value or keyword arguments")
            for k, v in kwargs.items():
                self.lock_value(k, v)
            return

        if isinstance(name, ParameterRef):
            name = str(name)
        if value == "null":
            value = self.pf.loc[name, "nullvalue"]
        # setting holdfast to 0 initially, so that the other value attributes are
        # allowed to be changed.
        self.pholdfast = {name: 0}
        self.pvals = {name: value}
        self.pinitvals = {name: value}
        # self.pnullvals = {name: value}
        self.pminimum = {name: value}
        self.pmaximum = {name: value}
        # finally, setting holdfast to 1.
        self.pholdfast = {name: 1}

    def pretty_table(self):
        self.unmangle()
        return self._parameter_bucket.pretty_table()

    def __p_rename(self, x):
        return str(self.rename_parameters.get(str(x), str(x)))

    def __repr__(self):
        s = "<larch."
        s += self.__class__.__name__
        try:
            is_mnl = self.is_mnl()
        except AttributeError:
            is_mnl = None
        else:
            if is_mnl:
                s += " (MNL)"
            else:
                s += " (GEV)"
        if self.title != "Untitled" and self.title is not None:
            s += f' "{self.title}"'
        s += ">"
        return s

    def initialize_graph(
        self, alternative_codes=None, alternative_names=None, root_id=0
    ):
        """
        Write a nesting tree graph for a MNL model.

        Parameters
        ----------
        alternative_codes : array-like, optional
            Explicitly give alternative codes. Ignored if `dataframes` is given
            or if the model has dataframes or a dataservice already set.
        alternative_names : array-like, optional
            Explicitly give alternative names. Ignored if `dataframes` is given
            or if the model has dataframes or a dataservice already set.
        root_id : int, default 0
            The id code of the root node.

        Raises
        ------
        ValueError
            The model is unable to infer the alternative codes to use.  This can
            be avoided by giving alternative codes explicitly or having previously
            set dataframes or a dataservice that will give the alternative codes.
        """
        if self.datatree is not None:

            def get_coords_array(*names):
                for name in names:
                    if name in self.datatree.root_dataset.coords:
                        return self.datatree.root_dataset.coords[name].values

            if alternative_codes is None:
                alternative_codes = get_coords_array(
                    self.datatree.ALTID,
                    "_altid_",
                    "altid",
                    "alt_id",
                    "alt_ids",
                    "alternative_id",
                    "alternative_ids",
                )
            if alternative_names is None:
                alternative_names = get_coords_array(
                    "altname",
                    "altnames",
                    "alt_name",
                    "alt_names",
                    "alternative_name",
                    "alternative_names",
                )

        if alternative_codes is None:
            return

        from .tree import NestingTree

        g = NestingTree(root_id=root_id)
        if alternative_names is None:
            for a in alternative_codes:
                g.add_node(a)
        else:
            for a, name in zip(alternative_codes, alternative_names):
                g.add_node(a, name=name)
        self._graph = g

    @property
    def graph(self) -> NestingTree:
        if self._graph is None:
            try:
                self.initialize_graph()
            except ValueError as err:
                import warnings

                warnings.warn(
                    "cannot initialize graph, must define alternatives somehow",
                    stacklevel=2,
                )
                raise RuntimeError(
                    "cannot initialize graph, must define alternatives somehow"
                ) from err
        return self._graph

    @graph.setter
    def graph(self, x: NestingTree | None):
        if x is not None and not isinstance(x, NestingTree):
            raise TypeError("graph must be a NestingTree")
        self._graph = x

    def utility_functions(self, subset=None, resolve_parameters=False):
        """
        Generate an XHTML output of the utility function(s).

        Parameters
        ----------
        subset : Collection, optional
            A collection of alternative codes to include. This only has effect if
            there are separate utility_co functions set by alternative. It is
            recommended to use this parameter if there are a very large number of
            alternatives, and the utility functions of most (or all) of them
            can be effectively communicated by showing only a few.
        resolve_parameters : bool, default False
            Whether to resolve the parameters to the current (estimated) value
            in the output.

        Returns
        -------
        xmle.Elem
        """
        self.unmangle()
        from xmle import Elem

        x = Elem("div")
        t = x.elem("table", style="margin-top:1px;", attrib={"class": "floatinghead"})
        if len(self.utility_co):
            # t.elem('caption', text=f"Utility Functions",
            # 	   style="caption-side:top;text-align:left;font-family:Roboto;font-weight:700;"
            # 			 "font-style:normal;font-size:100%;padding:0px;color:black;")

            # iterate over all alternatives if a dataframes is attached and lists the alternatives
            try:
                if self.dataservice is not None:
                    alts = self.dataservice.alternative_codes()
                elif self.dataframes is not None:
                    alts = self.dataframes.alternative_codes()
                else:
                    alts = self.utility_co.keys()
            except Exception:
                alts = self.utility_co.keys()
            t_head = t.elem("thead")
            tr = t_head.elem("tr")
            tr.elem("th", text="alt")
            tr.elem("th", text="formula", attrib={"style": "text-align:left;"})
            t_body = t.elem("tbody")
            for j in alts:
                if subset is None or j in subset:
                    tr = t_body.elem("tr")
                    tr.elem("td", text=str(j))
                    utilitycell = tr.elem("td", attrib={"style": "text-align:left;"})
                    utilitycell.elem("div")
                    anything = False
                    if len(self.utility_ca):
                        utilitycell[-1].tail = (utilitycell[-1].tail or "") + " + "
                        utilitycell << list(
                            self.utility_ca.__xml__(
                                linebreaks=True,
                                resolve_parameters=self,
                                value_in_tooltips=not resolve_parameters,
                            )
                        )
                        anything = True
                    if j in self.utility_co:
                        v = self.utility_co[j]
                        if len(v):
                            if anything:
                                utilitycell << Elem("br")
                            utilitycell[-1].tail = (utilitycell[-1].tail or "") + " + "
                            utilitycell << list(
                                v.__xml__(
                                    linebreaks=True,
                                    resolve_parameters=self,
                                    value_in_tooltips=not resolve_parameters,
                                )
                            )
                            anything = True
                    if len(self.quantity_ca):
                        if anything:
                            utilitycell << Elem("br")
                        if self.quantity_scale:
                            utilitycell[-1].tail = (utilitycell[-1].tail or "") + " + "
                            from .linear import ParameterRef

                            utilitycell << list(
                                ParameterRef(self.quantity_scale).__xml__(
                                    resolve_parameters=self,
                                    value_in_tooltips=not resolve_parameters,
                                )
                            )
                            utilitycell[-1].tail = (
                                utilitycell[-1].tail or ""
                            ) + " * log("
                        else:
                            utilitycell[-1].tail = (
                                utilitycell[-1].tail or ""
                            ) + " + log("
                        content = self.quantity_ca.__xml__(
                            linebreaks=True,
                            lineprefix="  ",
                            exponentiate_parameters=True,
                            resolve_parameters=self,
                            value_in_tooltips=not resolve_parameters,
                        )
                        utilitycell << list(content)
                        utilitycell.elem("br", tail=")")
        else:
            # there is no differentiation by alternatives, just give one formula
            # t.elem('caption', text=f"Utility Function",
            # 	   style="caption-side:top;text-align:left;font-family:Roboto;font-weight:700;"
            # 			 "font-style:normal;font-size:100%;padding:0px;color:black;")
            tr = t.elem("tr")
            utilitycell = tr.elem("td", attrib={"style": "text-align:left;"})
            utilitycell.elem("div")
            anything = False
            if len(self.utility_ca):
                utilitycell[-1].tail = (utilitycell[-1].tail or "") + " + "
                utilitycell << list(
                    self.utility_ca.__xml__(
                        linebreaks=True,
                        resolve_parameters=self,
                        value_in_tooltips=not resolve_parameters,
                    )
                )
                anything = True
            if len(self.quantity_ca):
                if anything:
                    utilitycell << Elem("br")
                if self.quantity_scale:
                    utilitycell[-1].tail = (utilitycell[-1].tail or "") + " + "
                    from .linear import ParameterRef

                    utilitycell << list(
                        ParameterRef(self.quantity_scale).__xml__(
                            resolve_parameters=self,
                            value_in_tooltips=not resolve_parameters,
                        )
                    )
                    utilitycell[-1].tail = (utilitycell[-1].tail or "") + " * log("
                else:
                    utilitycell[-1].tail = (utilitycell[-1].tail or "") + " + log("
                content = self.quantity_ca.__xml__(
                    linebreaks=True,
                    lineprefix="  ",
                    exponentiate_parameters=True,
                    resolve_parameters=self,
                    value_in_tooltips=not resolve_parameters,
                )
                utilitycell << list(content)
                utilitycell.elem("br", tail=")")
        return x

    def _utility_functions_as_frame(self, subset=None, resolve_parameters=False):
        """
        Generate a tabular output of the utility function(s).

        Parameters
        ----------
        subset : Collection, optional
            A collection of alternative codes to include. This only has effect if
            there are separate utility_co functions set by alternative. It is
            recommended to use this parameter if there are a very large number of
            alternatives, and the utility functions of most (or all) of them
            can be effectively communicated by showing only a few.
        resolve_parameters : bool, default False
            Whether to resolve the parameters to the current (estimated) value
            in the output.  Not implemented.

        Returns
        -------
        xmle.Elem
        """
        self.unmangle()

        tf = pd.DataFrame(
            index=pd.MultiIndex.from_tuples([], names=["Alt", "Line"]),
            columns=["Formula"],
        )

        if len(self.utility_co):
            # iterate over all alternatives if a dataframes is attached and lists the alternatives
            try:
                if self.dataservice is not None:
                    alts = self.dataservice.alternative_codes()
                elif self.dataframes is not None:
                    alts = self.dataframes.alternative_codes()
                else:
                    alts = self.utility_co.keys()
            except Exception:
                alts = self.utility_co.keys()

            for a in alts:
                if not (subset is None or a in subset):
                    continue
                line = 1
                op = " "
                for i in self.utility_ca:
                    tf.loc[(a, line), :] = f"{op} {str(i)}"
                    op = "+"
                    line += 1
                if a in self.utility_co:
                    for i in self.utility_co[a]:
                        tf.loc[(a, line), :] = f"{op} {str(i)}"
                        op = "+"
                        line += 1
                if len(self.quantity_ca):
                    if self.quantity_scale:
                        from .linear import ParameterRef

                        q = ParameterRef(self.quantity_scale)
                        tf.loc[(a, line), :] = f"{op} {str(q)} * log("
                    else:
                        tf.loc[(a, line), :] = f"{op} log("
                    op = " "
                    line += 1
                    for i in self.quantity_ca:
                        tf.loc[(a, line), :] = f"    {op} {i._str_exponentiate()}"
                        op = "+"
                        line += 1
                    tf.loc[(a, line), :] = ")"
                    op = "+"
                    line += 1

        else:
            # there is no differentiation by alternatives, just give one formula
            a = "*"
            line = 1
            op = " "
            for i in self.utility_ca:
                tf.loc[(a, line), :] = f"{op} {str(i)}"
                op = "+"
                line += 1
            if len(self.quantity_ca):
                if self.quantity_scale:
                    from .linear import ParameterRef

                    q = ParameterRef(self.quantity_scale)
                    tf.loc[(a, line), :] = f"{op} {str(q)} * log("
                else:
                    tf.loc[(a, line), :] = f"{op} log("
                op = " "
                line += 1
                for i in self.quantity_ca:
                    tf.loc[(a, line), :] = f"    {op} {i._str_exponentiate()}"
                    op = "+"
                    line += 1
                tf.loc[(a, line), :] = ")"
                op = "+"
                line += 1

        return tf

    def required_data(self):
        """
        Report what data is required in DataFrames for this model to be used.

        Returns
        -------
        dictx
        """
        try:
            from ..util import dictx

            req_data = dictx()

            if self.utility_ca is not None and len(self.utility_ca):
                if "ca" not in req_data:
                    req_data.ca = set()
                for i in self.utility_ca:
                    req_data.ca.add(str(i.data))

            if self.quantity_ca is not None and len(self.quantity_ca):
                if "ca" not in req_data:
                    req_data.ca = set()
                for i in self.quantity_ca:
                    req_data.ca.add(str(i.data))

            if self.utility_co is not None and len(self.utility_co):
                if "co" not in req_data:
                    req_data.co = set()
                for _alt, func in self.utility_co.items():
                    for i in func:
                        if str(i.data) != "1":
                            req_data.co.add(str(i.data))

            if "ca" in req_data:
                req_data.ca = list(sorted(req_data.ca))
            if "co" in req_data:
                req_data.co = list(sorted(req_data.co))

            if self.choice_ca_var:
                req_data.choice_ca = self.choice_ca_var
            elif self.choice_co_vars:
                req_data.choice_co = self.choice_co_vars
            elif self.choice_co_code:
                req_data.choice_co_code = self.choice_co_code
            elif self.choice_any:
                req_data.choice_any = True

            if self.weight_co_var:
                req_data.weight_co = self.weight_co_var

            if self.availability_ca_var:
                req_data.avail_ca = self.availability_ca_var
            elif self.availability_co_vars:
                req_data.avail_co = self.availability_co_vars
            elif self.availability_any:
                req_data.avail_any = True

            return req_data
        except Exception:
            logger.exception("error in required_data")

    def __contains__(self, item):
        return item in self.pf.index  # or (item in self.rename_parameters)

    @property
    def is_mangled(self):
        return self._mangled

    def mangle(self, data=True, structure=True) -> None:
        if data and not (self._mangled & 0x1):
            logger.debug(f"mangling data for {self.title}")
            self._mangled |= MANGLE_DATA
        if structure and not (self._mangled & MANGLE_STRUCTURE):
            logger.debug(f"mangling structure for {self.title}")
            self._mangled |= MANGLE_STRUCTURE

    def unmangle(self, force=False, structure_only=False):
        if not self._mangled and not force:
            return
        marker = f"_currently_unmangling_{__file__}"
        logger.debug(f"{marker=}")
        if getattr(self, marker, False):
            return
        try:
            setattr(self, marker, True)
            if force:
                self.mangle()
            logger.debug(f"{self._mangled=}")
            if (self._mangled & MANGLE_STRUCTURE) or force:
                self._scan_all_ensure_names()
                if structure_only:
                    self._mangled &= ~MANGLE_STRUCTURE
                else:
                    self._mangled = 0
        finally:
            delattr(self, marker)

    def _scan_all_ensure_names(self):
        self._scan_utility_ensure_names()
        self._scan_quantity_ensure_names()
        self._scan_logsums_ensure_names()
        self._scan_mixtures_ensure_names()

    def _scan_utility_ensure_names(self):
        """
        Scan the utility functions and ensure all named parameters appear in the parameter frame.

        Any named parameters that do not appear in the parameter frame are added.
        """
        nameset = set()
        u_co_dataset = set()
        for altcode, linear_function in self.utility_co.items():
            for component in linear_function:
                nameset.add(self.__p_rename(component.param))
                try:
                    u_co_dataset.add(str(component.data))
                except Exception:
                    warnings.warn(f"bad data in altcode {altcode}", stacklevel=2)
                    raise
        linear_function_ca = self.utility_ca
        for component in linear_function_ca:
            nameset.add(self.__p_rename(component.param))
        self._ensure_names(nameset)

    def _scan_quantity_ensure_names(self):
        if self.quantity_ca is not None:
            nameset = set()
            for component in self.quantity_ca:
                nameset.add(self.__p_rename(component.param))
            self._ensure_names(nameset)

    def _scan_logsums_ensure_names(self):
        nameset = set()
        if self._graph is not None:
            for nodecode in self._graph.topological_sorted_no_elementals:
                if nodecode != self._graph._root_id:
                    param_name = str(self._graph.nodes[nodecode]["parameter"])
                    nameset.add(self.__p_rename(param_name))
        if self.quantity_ca is not None and len(self.quantity_ca) > 0:
            if self.quantity_scale is not None:
                nameset.add(self.__p_rename(self.quantity_scale))
        if self.logsum_parameter is not None:
            nameset.add(self.__p_rename(self.logsum_parameter))
        self._ensure_names(
            nameset, value=1, nullvalue=1, initvalue=1, minimum=0.01, maximum=1
        )

    def _scan_mixtures_ensure_names(self):
        for i in self.mixtures:
            for name, default_value in i.param_names().items():
                self._ensure_names([name], value=default_value, initvalue=default_value)

    def _ensure_names(self, names, **kwargs):
        if self._parameter_bucket is not None:
            self._parameter_bucket.add_parameters(names, fill_values=kwargs)

    quantity_scale = SingleParameter()
    logsum_parameter = SingleParameter()

    def clear_cache(self):
        """Remove all cached log likelihood values and estimation results."""
        self._cached_loglike_best = None
        self._cached_loglike_null = None
        self._most_recent_estimation_result = None

    def _check_if_best(self, computed_ll, pvalues=None):
        if self._cached_loglike_best is None or computed_ll > self._cached_loglike_best:
            self._cached_loglike_best = computed_ll
            if pvalues is None:
                self._parameter_bucket._params = self._parameter_bucket._params.assign(
                    best=self._parameter_bucket._params["value"]
                )
            else:
                self._parameter_bucket._params = self._parameter_bucket._params.assign(
                    best=xr.DataArray(pvalues, dims=self._parameter_bucket.index_name)
                )

    @property
    def choice_ca_var(self):
        """Str : An |idca| variable giving the choices as indicator values."""
        return self._choice_ca_var

    @choice_ca_var.setter
    def choice_ca_var(self, x):
        if x is not None:
            x = str(x)
        if self._choice_ca_var != x:
            self.mangle()
        self._choice_ca_var = x
        if x is not None:
            self._choice_co_vars = None
            self._choice_co_code = None
            self._choice_any = False

    @property
    def choice_co_vars(self):
        """Dict[int,str] : A mapping giving |idco| expressions that evaluate to indicator values.

        Each key represents an alternative code number, and the associated expression
        gives the name of an |idco| variable or some function of |idco| variables that
        indicates whether that alternative was chosen.
        """
        if self._choice_co_vars:
            return self._choice_co_vars
        else:
            return None

    @choice_co_vars.setter
    def choice_co_vars(self, x):
        if isinstance(x, dict):
            if self._choice_co_vars != x:
                self.mangle()
            self._choice_co_vars = x
            self._choice_ca_var = None
            self._choice_co_code = None
            self._choice_any = False
        elif x is None:
            if self._choice_co_vars != x:
                self.mangle()
            self._choice_co_vars = x
        else:
            raise TypeError("choice_co_vars must be a dictionary")

    @choice_co_vars.deleter
    def choice_co_vars(self):
        self._choice_co_vars = None

    @property
    def choice_co_code(self):
        """Str : An |idco| variable giving the choices as alternative id's."""
        if self._choice_co_code:
            return self._choice_co_code
        else:
            return None

    @choice_co_code.setter
    def choice_co_code(self, x):
        if isinstance(x, str):
            if self._choice_co_code != x:
                self.mangle()
            self._choice_co_code = x
            self._choice_co_vars = None
            self._choice_ca_var = None
            self._choice_any = False
        elif x is None:
            if self._choice_co_code != x:
                self.mangle()
            self._choice_co_code = x
        else:
            raise TypeError("choice_co_vars must be a str")

    @choice_co_code.deleter
    def choice_co_code(self):
        if self._choice_co_code is not None:
            self.mangle()
        self._choice_co_code = None

    @property
    def choice_any(self):
        if self._choice_any:
            return True
        else:
            return False

    @choice_any.setter
    def choice_any(self, x):
        if x:
            self._choice_any = True
            self._choice_co_code = None
            self._choice_co_vars = None
            self._choice_ca_var = None
        else:
            self._choice_any = False

    @choice_any.deleter
    def choice_any(self):
        self._choice_any = False

    def choice_def(self, new_def: dict | None = None, **kwargs) -> dict:
        """
        Get or set the definition of the choice variable.

        This method can be used to check what the choice variable is currently
        set to, or to set the choice variable in a dynamic fashion.  The choice
        variable can be set in one of four ways:

        * choice_ca_var : A single |idca| variable or expression that evaluates
            to an indicator value
        * choice_co_vars : A dictionary of |idco| expressions, where each key is
            an alternative code and the value is a variable name or other
            expression that evaluates to an indicator value
        * choice_co_code : A single |idco| variable, which evaluates to the
            alternative code of the chosen alternative
        * choice_any : A flag for setting the choice to "True", which may used
            when the choice variable is not explicitly defined

        Returns
        -------
        dict
        """
        if len(kwargs) and new_def is None:
            new_def = kwargs
        if new_def is not None:
            if len(kwargs):
                raise ValueError("cannot give both new_def and other keyword args")
            if not isinstance(new_def, dict):
                raise TypeError("new choice definition must be given by a dictionary")
            if len(new_def) > 1:
                raise ValueError("new choice definition must have only one key")
            key, value = new_def.popitem()
            if key == "choice_ca_var":
                self.choice_ca_var = value
            elif key == "choice_co_vars":
                self.choice_co_vars = value
            elif key == "choice_co_code":
                self.choice_co_code = value
            elif key == "choice_any":
                self.choice_any = value
            else:
                raise ValueError("invalid key in new_def")

        if self._choice_ca_var:
            return {"choice_ca_var": self._choice_ca_var}
        elif self._choice_co_vars:
            return {"choice_co_vars": self._choice_co_vars}
        elif self._choice_co_code:
            return {"choice_co_code": self._choice_co_code}
        elif self._choice_any:
            return {"choice_any": True}
        else:
            raise ValueError("no choice variable defined")

    @property
    def weight_co_var(self):
        return self._weight_co_var

    @weight_co_var.setter
    def weight_co_var(self, x):
        if self._weight_co_var != x:
            self.mangle()
        self._weight_co_var = x

    @weight_co_var.deleter
    def weight_co_var(self):
        if self._weight_co_var is not None:
            self.mangle()
        self._weight_co_var = None

    @property
    def availability_ca_var(self):
        """Str : An |idca| variable or expression indicating if alternatives are available."""
        return self._availability_ca_var

    @availability_ca_var.setter
    def availability_ca_var(self, x):
        if x is not None:
            x = str(x)
        if self._availability_ca_var != x:
            self.mangle()
            self._availability_ca_var = x
            self._availability_co_vars = None
            self._availability_any = False

    @property
    def availability_co_vars(self):
        """Dict[int,str] : A mapping giving |idco| expressions that evaluate to availability indicators.

        Each key represents an alternative code number, and the associated expression
        gives the name of an |idco| variable or some function of |idco| variables that
        indicates whether that alternative is available.
        """
        if self._availability_co_vars:
            return self._availability_co_vars
        else:
            return None

    @availability_co_vars.setter
    def availability_co_vars(self, x):
        from collections.abc import Mapping

        if x is None:
            if self._availability_co_vars:
                self.mangle()
                self._availability_co_vars = {}
        else:
            if not isinstance(x, Mapping):
                raise TypeError(f"availability_co_vars must be dict not {type(x)}")
            if self._availability_co_vars != x:
                self.mangle()
            self._availability_co_vars = x
            self._availability_ca_var = None
            self._availability_any = False

    @property
    def availability_any(self):
        """Bool : A flag indicating whether availability should be inferred from the data.

        This only applies to DataFrames-based models, as the Dataset interface does
        not include a mechanism for the data to self-describe an availability feature.
        """
        return self._availability_any

    @availability_any.setter
    def availability_any(self, x):
        x = bool(x)
        if x != self._availability_any:
            self._availability_any = x
            self._availability_co_vars = None
            self._availability_ca_var = None

    @property
    def availability_var(self):
        raise NotImplementedError(
            "availability_var is no longer supported, use "
            "`availability_ca_var`, `availability_co_vars`, "
            "or `availability_any` instead."
        )

    def availability_def(self, new_def: dict | None = None, **kwargs) -> dict:
        """
        Get or set the definition of the availability variable.

        This method can be used to check what the availability variable is currently
        set to, or to set the availability variable in a dynamic fashion.  The
        availability variable can be set in one of three ways:

        * availability_ca_var : A single |idca| variable or expression that
            evaluates to an indicator value
        * availability_co_vars : A dictionary of |idco| expressions, where each
            key is an alternative code and the value is a variable name or other
            expression that evaluates to an indicator value
        * availability_any : A flag for setting availability to "True" for all
            alternatives, which may used when the availability variable is not
            otherwise explicitly defined

        Returns
        -------
        dict
        """
        if len(kwargs) and new_def is None:
            new_def = kwargs
        if new_def is not None:
            if len(kwargs):
                raise ValueError("cannot give both new_def and other keyword args")
            if not isinstance(new_def, dict):
                raise TypeError(
                    "new availability definition must be given by a dictionary"
                )
            if len(new_def) > 1:
                raise ValueError("new availability definition must have only one key")
            key, value = new_def.popitem()
            if key == "availability_ca_var":
                self.availability_ca_var = value
            elif key == "availability_co_vars":
                self.availability_co_vars = value
            elif key == "availability_any":
                self.availability_any = value
            else:
                raise ValueError("invalid key in new_def")

        if self._availability_ca_var:
            return {"availability_ca_var": self._availability_ca_var}
        elif self._availability_co_vars:
            return {"availability_co_vars": self._availability_co_vars}
        elif self._availability_any:
            return {"availability_any": True}
        else:
            raise ValueError("no availability variable defined")

    def parameter_summary(self):
        """
        Create a tabular summary of parameter values.

        This will generate a small table of parameters statistics,
        containing:

        *	Parameter Name (and Category, if applicable)
        *	Estimated Value
        *	Standard Error of the Estimate (if known)
        *	t Statistic (if known)
        *	Null Value
        *	Binding Constraints (if applicable)

        Returns
        -------
        pandas.DataFrame

        """
        NBSP = " "  # non=breaking space

        try:
            pbucket = self._parameter_bucket
        except AttributeError:
            pbucket = self

        tabledata = {}
        tabledata["Value"] = pbucket.pvals
        if "std_err" in pbucket._params:
            se = pbucket.pstderr
            tabledata["Std Err"] = se
            tstat = (pbucket.pvals - pbucket.pnullvals) / np.where(se, se, 1.0)
            tabledata["t Stat"] = np.where(se, tstat, np.nan)
        if "robust_std_err" in pbucket._params:
            se_ = pbucket._params["robust_std_err"]
            tabledata["Robust Std Err"] = se_
            tstat_ = (pbucket.pvals - pbucket.pnullvals) / np.where(se_, se_, 1.0)
            tabledata["Robust t Stat"] = np.where(se_, tstat_, np.nan)
        tabledata["Null Value"] = pbucket.pnullvals

        if "constrained" in pbucket.parameters:
            tabledata["Constrained"] = pbucket.parameters["constrained"]

        result = pd.DataFrame(tabledata, index=pbucket.pnames).rename_axis(
            index="Parameter"
        )

        # pf = self.pf
        # columns = [i for i in
        #           ['value', 'std_err', 't_stat', 'likelihood_ratio', 'nullvalue', 'constrained']
        #           if i in pf.columns]
        # result = pf[columns].rename(
        #     columns={
        #         'value': 'Value',
        #         'std_err': 'Std Err',
        #         't_stat': 't Stat',
        #         'likelihood_ratio': 'Like Ratio',
        #         'nullvalue': 'Null Value',
        #         'constrained': 'Constrained'
        #     }
        # )
        monospace_cols = []

        def fixie(x):
            if np.isfinite(x):
                if x > 1000:
                    return NBSP + "BIG"
                elif x < -1000:
                    return "-BIG"
                else:
                    return f"{x:0< 4.2f}".replace(" ", NBSP)
            else:
                return NBSP + "NA"

        if "t Stat" in result.columns:
            result.insert(result.columns.get_loc("t Stat") + 1, "Signif", "")
            result.loc[np.absolute(result["t Stat"]) > 1.9600, "Signif"] = "*"
            result.loc[np.absolute(result["t Stat"]) > 2.5758, "Signif"] = "**"
            result.loc[np.absolute(result["t Stat"]) > 3.2905, "Signif"] = "***"
            result["t Stat"] = result["t Stat"].apply(fixie)
            result.loc[result["t Stat"] == NBSP + "NA", "Signif"] = ""
            monospace_cols.append("t Stat")
            monospace_cols.append("Signif")
        if "Robust t Stat" in result.columns:
            result.insert(
                result.columns.get_loc("Robust t Stat") + 1, "Robust Signif", ""
            )
            result.loc[
                np.absolute(result["Robust t Stat"]) > 1.9600, "Robust Signif"
            ] = "*"
            result.loc[
                np.absolute(result["Robust t Stat"]) > 2.5758, "Robust Signif"
            ] = "**"
            result.loc[
                np.absolute(result["Robust t Stat"]) > 3.2905, "Robust Signif"
            ] = "***"
            result["Robust t Stat"] = result["Robust t Stat"].apply(fixie)
            result.loc[result["Robust t Stat"] == NBSP + "NA", "Robust Signif"] = ""
            monospace_cols.append("Robust t Stat")
            monospace_cols.append("Robust Signif")
        if "Like Ratio" in result.columns:
            if "Signif" not in result.columns:
                result.insert(result.columns.get_loc("Like Ratio") + 1, "Signif", "")
            if "t Stat" in result.columns:
                non_finite_t = ~np.isfinite(result["t Stat"])
            else:
                non_finite_t = True
            result.loc[
                np.absolute((np.isfinite(result["Like Ratio"])) & non_finite_t),
                "Signif",
            ] = "[]"
            result.loc[
                np.absolute(((result["Like Ratio"]) > 1.9207) & non_finite_t), "Signif"
            ] = "[*]"
            result.loc[
                np.absolute(((result["Like Ratio"]) > 3.3174) & non_finite_t), "Signif"
            ] = "[**]"
            result.loc[
                np.absolute(((result["Like Ratio"]) > 5.4138) & non_finite_t), "Signif"
            ] = "[***]"
            result["Like Ratio"] = result["Like Ratio"].apply(fixie)
            monospace_cols.append("Like Ratio")
            if "Signif" not in monospace_cols:
                monospace_cols.append("Signif")
        for z in ["Std Err", "Robust Std Err"]:
            if z in result.columns:
                _fmt_s = (
                    lambda x: f"{x: #.3g}".replace(" ", NBSP)
                    if np.isfinite(x)
                    else NBSP + "NA"
                )
                result[z] = result[z].apply(_fmt_s)
                monospace_cols.append(z)
        if "Value" in result.columns:
            result["Value"] = result["Value"].apply(
                lambda x: f"{x: #.3g}".replace(" ", NBSP)
            )
            monospace_cols.append("Value")
        if "Constrained" in result.columns:
            result["Constrained"] = result["Constrained"].str.replace("\n", "<br/>")
        if "Null Value" in result.columns:
            monospace_cols.append("Null Value")
        # if result.index.nlevels > 1:
        #     pnames = result.index.get_level_values(-1)
        # else:
        #     pnames = result.index
        styles = [
            dict(
                selector="th",
                props=[
                    ("vertical-align", "top"),
                    ("text-align", "left"),
                ],
            ),
            dict(
                selector="td",
                props=[
                    ("vertical-align", "top"),
                    ("text-align", "left"),
                ],
            ),
        ]

        if self.ordering:
            paramset = set(result.index)
            out = []
            import re

            for category in self.ordering:
                category_name = category[0]
                category_params = []
                for category_pattern in category[1:]:
                    category_params.extend(
                        sorted(
                            i
                            for i in paramset
                            if re.match(category_pattern, i) is not None
                        )
                    )
                    paramset -= set(category_params)
                out.append([category_name, category_params])
            if len(paramset):
                out.append(["Other", sorted(paramset)])

            tuples = []
            for c, pp in out:
                for p in pp:
                    tuples.append((c, p))

            ix = pd.MultiIndex.from_tuples(tuples, names=["Category", "Parameter"])

            result = result.reindex(ix.get_level_values(1))
            result.index = ix

        return (
            result.style.set_table_styles(styles)
            .format({"Null Value": "{: .2f}"})
            .apply(
                lambda x: ["font-family:monospace" for _ in x],
                axis="columns",
                subset=monospace_cols,
            )
            # newer versions of pandas will allow the following instead of the above line:
            # .map(lambda x: "font-family:monospace", subset=monospace_cols)
        )

    def estimation_statistics(self, compute_loglike_null=True):
        """
        Create an XHTML summary of estimation statistics.

        This will generate a small table of estimation statistics,
        containing:

        *	Log Likelihood at Convergence
        *	Log Likelihood at Null Parameters (if known)
        *	Log Likelihood with No Model (if known)
        *	Log Likelihood at Constants Only (if known)

        Additionally, for each included reference value (i.e.
        everything except log likelihood at convergence) the
        rho squared with respect to that value is also given.

        Each statistic is reported in aggregate, as well as
        per case.

        Parameters
        ----------
        compute_loglike_null : bool, default True
            If the log likelihood at null values has not already
            been computed (i.e., if it is not cached) then compute
            it, cache its value, and include it in the output.

        Returns
        -------
        xmle.Elem

        """
        from xmle import Elem

        div = Elem("div")
        table = div.put("table")

        thead = table.put("thead")
        tr = thead.put("tr")
        tr.put("th", text="Statistic")
        tr.put("th", text="Aggregate")
        tr.put("th", text="Per Case")

        tbody = table.put("tbody")

        try:
            ncases = self.n_cases
        except (MissingDataError, AttributeError):
            ncases = None

        tr = tbody.put("tr")
        tr.put("td", text="Number of Cases")
        if ncases:
            tr.put("td", text=str(ncases), colspan="2")
        else:
            tr.put("td", text="not available", colspan="2")

        mostrecent = self._most_recent_estimation_result
        if mostrecent is not None:
            tr = tbody.put("tr")
            tr.put("td", text="Log Likelihood at Convergence")
            tr.put("td", text=f"{mostrecent.loglike:.2f}")
            if ncases:
                tr.put("td", text=f"{mostrecent.loglike / ncases:.2f}")
            else:
                tr.put("td", text="na")

        ll_z = self._cached_loglike_null
        if ll_z == 0 or ll_z is None:
            if compute_loglike_null:
                try:
                    ll_z = self.loglike_null()
                except (MissingDataError, AttributeError):
                    ll_z = 0
            else:
                ll_z = 0
        if ll_z:
            tr = tbody.put("tr")
            tr.put("td", text="Log Likelihood at Null Parameters")
            tr.put("td", text=f"{ll_z:.2f}")
            if ncases:
                tr.put("td", text=f"{ll_z / ncases:.2f}")
            else:
                tr.put("td", text="na")
            if mostrecent is not None:
                tr = tbody.put("tr")
                tr.put("td", text="Rho Squared w.r.t. Null Parameters")
                rsz = 1.0 - (mostrecent.loglike / ll_z)
                tr.put("td", text=f"{rsz:.3f}", colspan="2")

        try:
            ll_nil = self._cached_loglike_nil
        except (MissingDataError, AttributeError):
            ll_nil = 0
        if ll_nil:
            tr = tbody.put("tr")
            tr.put("td", text="Log Likelihood with No Model")
            tr.put("td", text=f"{ll_nil:.2f}")
            if ncases:
                tr.put("td", text=f"{ll_nil / ncases:.2f}")
            else:
                tr.put("td", text="na")
            if mostrecent is not None:
                tr = tbody.put("tr")
                tr.put("td", text="Rho Squared w.r.t. No Model")
                rsz = 1.0 - (mostrecent.loglike / ll_nil)
                tr.put("td", text=f"{rsz:.3f}", colspan="2")

        try:
            ll_c = self._cached_loglike_constants_only
        except (MissingDataError, AttributeError):
            ll_c = 0
        if ll_c:
            tr = tbody.put("tr")
            tr.put("td", text="Log Likelihood at Constants Only")
            tr.put("td", text=f"{ll_c:.2f}")
            if ncases:
                tr.put("td", text=f"{ll_c / ncases:.2f}")
            else:
                tr.put("td", text="na")
            if mostrecent is not None:
                tr = tbody.put("tr")
                tr.put("td", text="Rho Squared w.r.t. Constants Only")
                rsc = 1.0 - (mostrecent.loglike / ll_c)
                tr.put("td", text=f"{rsc:.3f}", colspan="2")

        return div

    def estimation_statistics_raw(self, compute_loglike_null=True) -> pd.Series:
        """Compile estimation statistics as a pandas Series."""
        stats = {}

        try:
            ncases = self.n_cases
        except (MissingDataError, AttributeError):
            ncases = None

        if ncases:
            stats[("Number of Cases", "Aggregate")] = ncases
        else:
            stats[("Number of Cases", "Aggregate")] = "not available"

        mostrecent = self._most_recent_estimation_result
        if mostrecent is not None:
            stats[("Log Likelihood at Convergence", "Aggregate")] = mostrecent.loglike
            if ncases:
                stats[("Log Likelihood at Convergence", "Per Case")] = (
                    mostrecent.loglike / ncases
                )

        ll_z = self._cached_loglike_null
        if ll_z == 0 or ll_z is None:
            if compute_loglike_null:
                try:
                    ll_z = self.loglike_null()
                except (MissingDataError, AttributeError):
                    ll_z = 0
            else:
                ll_z = 0
        if ll_z:
            stats[("Log Likelihood at Null Parameters", "Aggregate")] = ll_z
            if ncases:
                stats[("Log Likelihood at Null Parameters", "Per Case")] = ll_z / ncases
            if mostrecent is not None:
                rsz = 1.0 - (mostrecent.loglike / ll_z)
                stats[("Rho Squared w.r.t. Null Parameters", "Aggregate")] = rsz

        try:
            ll_nil = self._cached_loglike_nil
        except (MissingDataError, AttributeError):
            ll_nil = 0
        if ll_nil:
            stats[("Log Likelihood with No Model", "Aggregate")] = ll_nil
            if ncases:
                stats[("Log Likelihood with No Model", "Per Case")] = ll_nil / ncases
            if mostrecent is not None:
                rsz = 1.0 - (mostrecent.loglike / ll_nil)
                stats[("Rho Squared w.r.t. No Model", "Aggregate")] = rsz

        try:
            ll_c = self._cached_loglike_constants_only
        except (MissingDataError, AttributeError):
            ll_c = 0
        if ll_c:
            stats[("Log Likelihood at Constants Only", "Aggregate")] = ll_c
            if ncases:
                stats[("Log Likelihood at Constants Only", "Per Case")] = ll_c / ncases
            if mostrecent is not None:
                rsc = 1.0 - (mostrecent.loglike / ll_c)
                stats[("Rho Squared w.r.t. Constants Only", "Aggregate")] = rsc

        return pd.Series(stats)

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, TypeVar

import numba as nb
import numpy as np
import pandas as pd
import xarray as xr

from larch.warning import WeightRescaleWarning

from .construct import _DatasetConstruct
from .dim_names import ALTID, ALTIDX, CASEALT, CASEID, CASEPTR, GROUPID, INGROUP

if TYPE_CHECKING:
    import sharrow as sh
    from numpy.typing import ArrayLike

    from larch.dataset import DataTree

DataT = TypeVar("DataT", bound=xr.Dataset | xr.DataArray)


@nb.njit
def case_ptr_to_indexes(n_casealts, case_ptrs):
    case_index = np.zeros(n_casealts, dtype=np.int64)
    for c in range(case_ptrs.shape[0] - 1):
        case_index[case_ptrs[c] : case_ptrs[c + 1]] = c
    return case_index


@nb.njit
def any_row_sums_to_not_one(arr):
    for i in range(arr.shape[0]):
        row_sum = 0
        for j in range(arr.shape[1]):
            row_sum += arr[i, j]
        if row_sum < 0.99999999 or row_sum > 1.00000001:
            return True
    return False


@nb.njit
def ce_dissolve_zero_variance(ce_data, ce_caseptr):
    """
    Dissolves zero variance in the given data based on case pointers.

    Parameters
    ----------
    ce_data : array-like, shape [n_casealts]
        One-dimensional array containing the data.
    ce_caseptr : array-like, shape [n_cases]
        One-dimensional array containing the case pointers.

    Returns
    -------
    out : ndarray
        One-dimensional array containing the dissolved data.
    flag : int
        Flag indicating whether variance was detected or not.
        - 1 if variance was detected
        - 0 if no variance was found and the `out` array is valid.
    """
    failed = 0
    if ce_caseptr.ndim == 2:
        ce_caseptr1 = ce_caseptr[:, -1]
    else:
        ce_caseptr1 = ce_caseptr[1:]
    shape = (ce_caseptr1.shape[0],)
    out = np.zeros(shape, dtype=ce_data.dtype)
    c = 0
    out[0] = ce_data[0]
    for row in range(ce_data.shape[0]):
        if row == ce_caseptr1[c]:
            c += 1
            out[c] = ce_data[row]
        else:
            if out[c] != ce_data[row]:
                failed = 1
                break
    return out, failed


class _GenericFlow:
    def __init__(self, x: xr.DataArray | xr.Dataset | None = None):
        self._obj = x
        self._flow_library = {}

    @property
    def CASEID(self) -> str | None:
        """Str : The _caseid_ dimension of this Dataset, if defined."""
        result = self._obj.attrs.get(CASEID, None)
        return result

    def __set_attr(self, attr_name, target_value, check_dim=True):
        if target_value is None:
            if attr_name in self._obj.attrs:
                del self._obj.attrs[attr_name]
        else:
            if check_dim and target_value not in self._obj.dims:
                raise ValueError(f"cannot set {attr_name}, {target_value} not in dims")
            self._obj.attrs[attr_name] = target_value

    @CASEID.setter
    def CASEID(self, dim_name):
        self.__set_attr(CASEID, dim_name, check_dim=self.GROUPID is not None)

    @property
    def ALTID(self) -> str | None:
        """The _altid_ dimension of this Dataset, if defined."""
        result = self._obj.attrs.get(ALTID, None)
        return result

    @ALTID.setter
    def ALTID(self, dim_name):
        self.__set_attr(ALTID, dim_name)

    @property
    def CASEALT(self) -> str | None:
        """The _casealt_ dimension of this Dataset, if defined."""
        result = self._obj.attrs.get(CASEALT, None)
        return result

    @CASEALT.setter
    def CASEALT(self, dim_name):
        self.__set_attr(CASEALT, dim_name)

    @property
    def ALTIDX(self) -> str | None:
        """Str : The _alt_idx_ dimension of this Dataset, if defined."""
        result = self._obj.attrs.get(ALTIDX, None)
        return result

    @ALTIDX.setter
    def ALTIDX(self, dim_name):
        self.__set_attr(ALTIDX, dim_name, False)

    @property
    def CASEPTR(self) -> str | None:
        """The _caseptr_ dimension of this Dataset, if defined."""
        result = self._obj.attrs.get(CASEPTR, None)
        return result

    @CASEPTR.setter
    def CASEPTR(self, dim_name):
        self.__set_attr(CASEPTR, dim_name, False)

    @property
    def GROUPID(self) -> str | None:
        """The _groupid_ dimension of this Dataset, if defined."""
        result = self._obj.attrs.get(GROUPID, None)
        return result

    @GROUPID.setter
    def GROUPID(self, dim_name):
        self.__set_attr(GROUPID, dim_name)

    @property
    def INGROUP(self) -> str | None:
        """The _ingroup_ dimension of this Dataset, if defined."""
        result = self._obj.attrs.get(INGROUP, None)
        return result

    @INGROUP.setter
    def INGROUP(self, dim_name):
        self.__set_attr(INGROUP, dim_name)

    def set_altids(
        self, altids, dim_name=None, dtype=None, inplace=False
    ) -> xr.Dataset:
        """
        Set the alternative ids for this Dataset.

        Parameters
        ----------
        altids : array-like of int
            Integer id codes.
        dim_name : str, optional
            Use this dimension name for alternatives.
        dtype : dtype, optional
            Coerce the altids to be this dtype.
        inplace : bool, default False
            When true, apply the transformation in-place on the Dataset,
            otherwise return a modified copy.

        Returns
        -------
        xarray.Dataset
            The modified Dataset.

        Examples
        --------
        >>> import numpy as np
        >>> from larch.dataset import Dataset
        >>> ds = Dataset()
        >>> ds = ds.dc.set_altids([1,2,3,4], dtype=np.int64)
        >>> ds.dc.altids()
        Int64Index([1, 2, 3, 4], dtype='int64', name='_altid_')
        >>> ds.dc.n_alts
        4
        >>> ds2 = ds.dc.set_altids([7,8,9], dim_name='A', dtype=np.int64)
        >>> ds2.dc.ALTID
        'A'
        >>> ds2.dc.altids()
        Int64Index([7, 8, 9], dtype='int64', name='A')
        >>> ds2
        <xarray.Dataset>
        Dimensions:  (_altid_: 4, A: 3)
        Coordinates:
          * _altid_  (_altid_) int64 1 2 3 4
          * A        (A) int64 7 8 9
        Data variables:
            *empty*
        Attributes:
            _altid_:  A
        """
        if inplace:
            obj = self._obj
        else:
            obj = self._obj.copy()
        dim_name = (
            dim_name or getattr(altids, "name", None) or obj.attrs.get(ALTID, ALTID)
        )
        if not isinstance(altids, xr.DataArray):
            altids = xr.DataArray(
                np.asarray(altids),
                dims=(dim_name),
            )
        if dtype is not None:
            altids = altids.astype(dtype)
        obj.coords[dim_name] = altids
        obj.dc.ALTID = dim_name
        return obj

    def set_altnames(self, altnames, inplace=False) -> xr.Dataset:
        """
        Set the alternative names for this Dataset.

        Parameters
        ----------
        altnames : Mapping or array-like
            A mapping of (integer) codes to names, or an array or names
            of the same length and order as the alternatives already
            defined in this Dataset.
        inplace : bool, default False
            When true, apply the transformation in-place on the Dataset,
            otherwise return a modified copy.

        Returns
        -------
        xarray.Dataset
            The modified Dataset.  A value is returned regardless of whether
            `inplace` is True.
        """
        if inplace:
            obj = self._obj
        else:
            obj = self._obj.copy()
        if isinstance(altnames, Mapping):
            a = obj.dc.ALTID
            names = xr.DataArray(
                [altnames.get(i, None) for i in obj[a].values],
                dims=a,
            )
        elif isinstance(altnames, xr.DataArray):
            names = altnames
        else:
            if obj.dc.ALTID is None:
                obj = obj.dc.set_altids(np.arange(1, len(altnames) + 1))
            names = xr.DataArray(
                np.asarray(altnames),
                dims=obj.dc.ALTID,
            )
        obj.coords["alt_names"] = names
        return obj

    def as_tree(self, label="main", exclude_dims=(), extra_vars=None) -> DataTree:
        """
        Convert this Dataset to a DataTree.

        For |idco| and |idca| datasets, the result will generally be a
        single-node tree.  For |idce| data, there will be

        Parameters
        ----------
        label : str, default 'main'
            Name to use for the root node in the tree.
        exclude_dims : Tuple[str], optional
            Exclude these dimensions, in addition to any
            dimensions listed in the `_exclude_dims_` attribute.
        extra_vars : dict, optional
            Extra variables to add to the tree, which can be referenced
            in expressions.

        Returns
        -------
        DataTree
        """
        from ..dataset import DataTree

        if self.CASEPTR is not None:
            case_index = case_ptr_to_indexes(
                self._obj.sizes[self.CASEALT], self[self.CASEPTR].values
            )
            obj = self._obj.assign(
                {"_case_index_": xr.DataArray(case_index, dims=(self.CASEALT))}
            )
            tree = DataTree(
                **{label: obj.drop_dims(self.CASEID)}, extra_vars=extra_vars
            )
            ds = obj.keep_dims(self.CASEID)
            ds.attrs.pop("_exclude_dims_", None)
            ds.attrs.pop("_caseptr_", None)
            ds.attrs.pop("_casealt_", None)
            ds.attrs.pop("_alt_idx_", None)
            tree.add_dataset(
                "idcoVars",
                ds,
                relationships=(f"{label}._case_index_ -> idcoVars.{self.CASEID}"),
            )
        else:
            tree = DataTree(**{label: self._obj}, extra_vars=extra_vars)
        return tree

    def setup_flow(self, *args, **kwargs) -> sh.Flow:
        """
        Set up a new Flow for analysis using the structure of this DataTree.

        This method creates a new `DataTree` with only this Dataset as
        the root Dataset labeled `main`.  All other arguments are passed
        through to `DataTree.setup_flow`.

        Returns
        -------
        Flow
        """
        return self.as_tree().setup_flow(*args, **kwargs)

    def caseids(self) -> pd.Index:
        """
        Access the caseids coordinates as an index.

        Returns
        -------
        pandas.Index
        """
        return self._obj.indexes[self.CASEID]

    def altids(self) -> pd.Index:
        """
        Access the altids coordinates as an index.

        Returns
        -------
        pandas.Index
        """
        return self._obj.indexes[self.ALTID]

    def groupids(self) -> pd.Index:
        """
        Access the groupids coordinates as an index.

        Returns
        -------
        pandas.Index
        """
        return self._obj.indexes[self.GROUPID]

    @property
    def alts_mapping(self) -> dict[int, str]:
        """Mapping of alternative codes to names."""
        a = self._obj.coords[self.ALTID]
        if "alt_names" in a.coords:
            return dict(zip(a.values, a.coords["alt_names"].values))
        else:
            return dict(zip(a.values, a.values))

    @property
    def n_cases(self) -> int:
        """The number of discrete choice cases in this Dataset."""
        try:
            return self._obj.sizes[self.CASEID]
        except KeyError:
            try:
                return self._obj.sizes[self.GROUPID] * self._obj.sizes[self.INGROUP]
            except KeyError:
                pass
            logging.getLogger().error(
                f"missing {self.CASEID!r} among dims {self._obj.sizes}"
            )
            raise

    @property
    def n_panels(self):
        try:
            return self._obj.sizes[self.GROUPID]
        except KeyError:
            try:
                return self._obj.sizes[self.CASEID]
            except KeyError:
                pass
            logging.getLogger().error(
                f"missing {self.GROUPID!r} and {self.CASEID!r} among dims {self._obj.sizes}"
            )
            raise

    @property
    def n_in_panel(self):
        return self._obj.sizes[self.INGROUP]

    def transfer_dimension_attrs(self, target: DataT) -> DataT:
        """
        Transfer discrete choice dimension attributes to a new Dataset or DataArray.

        Parameters
        ----------
        target : xarray.DataArray or xarray.Dataset

        Returns
        -------
        xarray.DataArray or xarray.Dataset
        """
        if not isinstance(target, xr.DataArray | xr.Dataset):
            return target
        updates = {}
        for i in [CASEID, ALTID, GROUPID, INGROUP, CASEALT, CASEPTR]:
            j = self._obj.attrs.get(i, None)
            if j is not None and j in target.dims:
                updates[i] = j
        return target.assign_attrs(updates)

    def get_expr(self, expression) -> xr.DataArray:
        """
        Access or evaluate an expression.

        Parameters
        ----------
        expression : str

        Returns
        -------
        xarray.DataArray
        """
        try:
            result = self._obj[expression]
        except (KeyError, IndexError):
            if expression in self._flow_library:
                flow = self._flow_library[expression]
            else:
                flow = self.setup_flow({expression: expression})
                self._flow_library[expression] = flow
            if flow.tree.root_dataset is not self:
                flow.tree = self.as_tree()
            result = flow.load_dataarray().isel(expressions=0)
        result = self.transfer_dimension_attrs(result)
        return result

    def dissolve_zero_variance(self, dim="<ALTID>", inplace=False) -> xr.Dataset:
        """
        Dissolve dimension on variables where it has no variance.

        This method is convenient to convert variables that have
        been loaded as |idca| or |idce| format into |idco| format where
        appropriate.

        Parameters
        ----------
        dim : str, optional
            The name of the dimension to potentially dissolve.
        inplace : bool, default False
            Whether to dissolve variables in-place.

        Returns
        -------
        xarray.Dataset
            The modified Dataset. A value is returned regardless of whether
            `inplace` is True.
        """
        if dim == "<ALTID>":
            dim = self.ALTID
        if inplace:
            obj = self
        else:
            obj = self.copy()
        for k in obj.variables:
            if obj[k].dtype.kind in {"U", "S", "O"}:
                continue
            if dim in obj[k].dims:
                try:
                    dissolve = obj[k].std(dim=dim).max() < 1e-10
                except TypeError:
                    pass
                else:
                    if dissolve:
                        obj[k] = obj[k].min(dim=dim)
            elif obj[k].dims == (self.CASEALT,):
                proposal, flag = ce_dissolve_zero_variance(
                    obj[k].values, obj[self.CASEPTR].values
                )
                if flag == 0:
                    obj = obj.assign({k: xr.DataArray(proposal, dims=(self.CASEID))})
        return obj

    def dissolve_coords(self, dim, others=None):
        d = self._obj.reset_index(dim)
        a = d[f"{dim}_"]
        mapper = dict((j, i) for (i, j) in enumerate(a.to_series()))
        mapper_f = np.vectorize(mapper.get)
        if others is None:
            others = []
        if isinstance(others, str):
            others = [others]
        for other in others:
            d[other] = xr.apply_ufunc(mapper_f, d[other])
        return d


@xr.register_dataarray_accessor("dc")
class _DataArrayDC(_GenericFlow):
    """Larch discrete choice attributes and methods for xarray.DataArray."""

    _parent_class = xr.DataArray

    @property
    def n_alts(self):
        """The number of discrete choice alternatives in this DataArray."""
        if self.ALTID in self._obj.sizes:
            return self._obj.sizes.get(self.ALTID)
        if "n_alts" in self._obj.attrs:
            return self._obj.attrs["n_alts"]
        if self.name:
            raise ValueError(f"no n_alts set for {self.name!r}")
        raise ValueError("no n_alts set")

    def __getitem__(self, name):
        # pass dimension attrs to DataArray
        result = self._obj[name]
        result = self.transfer_dimension_attrs(result)
        return result

    def __getattr__(self, name):
        # pass dimension attrs to DataArray
        result = getattr(self._obj, name)
        result = self.transfer_dimension_attrs(result)
        return result


@xr.register_dataset_accessor("dc")
class _DatasetDC(_GenericFlow):
    """Larch discrete choice attributes and methods for xarray.Dataset."""

    _parent_class = xr.Dataset

    @property
    def n_alts(self) -> int:
        """The number of discrete choice alternatives in this Dataset."""
        if self.ALTID in self._obj.sizes:
            return self._obj.sizes[self.ALTID]
        if "n_alts" in self._obj.attrs:
            return self._obj.attrs["n_alts"]
        raise ValueError("no n_alts set")

    def __getitem__(self, name):
        # pass dimension attrs to DataArray
        result = self._obj[name]
        result = self.transfer_dimension_attrs(result)
        return result

    def __getattr__(self, name):
        # pass dimension attrs to DataArray
        result = getattr(self._obj, name)
        result = self.transfer_dimension_attrs(result)
        return result

    def __contains__(self, item):
        return self._obj.__contains__(item)

    def query_cases(self, query, parser="pandas", engine=None):
        """
        Return a new dataset with each array indexed along the CASEID dimension.

        The indexers are given as strings containing Python expressions to be
        evaluated against the data variables in the dataset.

        Parameters
        ----------
        query : str
            Python expressions to be evaluated against the data variables
            in the dataset. The expressions will be evaluated using the pandas
            eval() function, and can contain any valid Python expressions but cannot
            contain any Python statements.
        parser : {"pandas", "python"}, default: "pandas"
            The parser to use to construct the syntax tree from the expression.
            The default of 'pandas' parses code slightly different than standard
            Python. Alternatively, you can parse an expression using the 'python'
            parser to retain strict Python semantics.
        engine : {"python", "numexpr", None}, default: None
            The engine used to evaluate the expression. Supported engines are:

            - None: tries to use numexpr, falls back to python
            - "numexpr": evaluates expressions using numexpr
            - "python": performs operations as if you had eval’d in top level python

        Returns
        -------
        obj : Dataset
            A new Dataset with the same contents as this dataset, except each
            array is indexed by the results of the query on the CASEID dimension.

        See Also
        --------
        Dataset.isel
        pandas.eval
        """
        result = self._obj.query({self.CASEID: query}, parser=parser, engine=engine)
        result = self.transfer_dimension_attrs(result)
        return result

    def to_arrays(self, graph, float_dtype=np.float64):
        from ..model.cascading import array_av_cascade, array_ch_cascade
        from ..model.data_arrays import DataArrays

        if "co" in self:
            co = self["co"].values.astype(float_dtype)
        else:
            co = np.empty((self.n_cases, 0), dtype=float_dtype)

        if "ca" in self:
            ca = self["ca"].values.astype(float_dtype)
        else:
            ca = np.empty((self.n_cases, self.n_alts, 0), dtype=float_dtype)

        if "ce_data" in self:
            ce_data = self["ce_data"].values.astype(float_dtype)
        else:
            ce_data = np.empty((0, 0), dtype=float_dtype)

        if self.ALTIDX is not None:
            ce_altidx = self[self.ALTIDX].values
        else:
            ce_altidx = np.empty((0), dtype=np.int16)

        if self.CASEPTR is not None:
            ce_caseptr = np.lib.stride_tricks.sliding_window_view(
                self[self.CASEPTR].values, 2
            )
        else:
            ce_caseptr = np.empty((self.n_cases, 0), dtype=np.int16)

        if "wt" in self:
            wt = self["wt"].values.astype(float_dtype)
        else:
            if self.GROUPID is None:
                wt = np.ones(self.n_cases, dtype=float_dtype)
            else:
                wt = np.ones(self.n_panels, dtype=float_dtype)

        if "ch" in self:
            ch = array_ch_cascade(self["ch"].values, graph, dtype=float_dtype)
        else:
            if self.GROUPID is None:
                ch = np.zeros([self.n_cases, len(graph)], dtype=float_dtype)
            else:
                ch = np.zeros(
                    [self.n_panels, self.n_in_panel, len(graph)], dtype=float_dtype
                )

        if "av" in self:
            av = array_av_cascade(self["av"].values, graph)
        else:
            if self.GROUPID is None:
                av = np.ones([self.n_cases, len(graph)], dtype=np.int8)
            else:
                av = np.ones(
                    [self.n_panels, self.n_in_panel, len(graph)], dtype=np.int8
                )

        return DataArrays(ch, av, wt, co, ca, ce_data, ce_altidx, ce_caseptr)

    def autoscale_weights(self) -> float:
        """
        Scale the weights so the average weight is 1.

        If weights are embedded in the choice variable,
        they are extracted (so the total choice in each case
        is 0.0 or 1.0, and the weight is isolated in the
        data_wt terms) before any scaling is applied.

        Returns
        -------
        scale : float
        """
        need_to_extract_wgt_from_ch = any_row_sums_to_not_one(self._obj["ch"].values)

        if need_to_extract_wgt_from_ch and "wt" not in self._obj:
            self._obj["wt"] = xr.DataArray(
                data=1.0,
                coords={self.CASEID: self._obj[self.CASEID]},
                dims=(self.CASEID),
            )

        if need_to_extract_wgt_from_ch:
            wgt_from_ch = self._obj["ch"].sum(self.ALTID)
            wgt_from_ch = wgt_from_ch.where(wgt_from_ch > 0, 1)
            self._obj["ch"] /= wgt_from_ch
            self._obj["wt"] *= wgt_from_ch

        if "wt" not in self._obj:
            return 1.0

        total_weight = self._obj["wt"].sum()
        scale_level = float(total_weight) / self.n_cases

        normalization = self._obj["wt"].attrs.get("normalization", 1.0)
        normalization *= scale_level
        self._obj["wt"].attrs["normalization"] = normalization

        if scale_level != 0:
            self._obj["wt"] /= scale_level

        if scale_level < 0.99999 or scale_level > 1.00001:
            warnings.warn(
                f"rescaled array of weights by a factor of {scale_level}",
                WeightRescaleWarning,
                stacklevel=2,
            )

        return scale_level

    def unscale_weights(self) -> float:
        """
        Unscale the weights, restoring the original values.

        This method is the inverse of `autoscale_weights`.

        Returns
        -------
        scale : float
        """
        scale_level = self._obj["wt"].attrs.get("normalization", 1.0)
        if scale_level != 0:
            self._obj["wt"] *= scale_level
        self._obj["wt"].attrs["normalization"] = 1.0
        return scale_level

    @staticmethod
    def from_idco(df, alts: Mapping[int, str] | ArrayLike[int] | None = None):
        """
        Construct a Dataset from an idco-format DataFrame.

        Parameters
        ----------
        df : DataFrame
            The input data should be an idco-format DataFrame, with
            the caseid's in a single-level index,
        alts : Mapping or array-like, optional
            If given as a mapping, links alternative codes to names.
            An array or list of integers gives codes for the alternatives,
            which are otherwise unnamed.

        Returns
        -------
        Dataset
        """
        return _DatasetConstruct.from_idco(df, alts=alts)

    @staticmethod
    def from_idca(
        df: pd.DataFrame,
        *,
        crack: bool = True,
        altnames: Mapping[int, str] | Sequence[str] = None,
        avail: str = "_avail_",
        fill_missing: dict = None,
    ) -> xr.Dataset:
        """
        Construct a Dataset from an idca-format DataFrame.

        This method loads the data as dense arrays.

        Parameters
        ----------
        df : DataFrame
            The input data should be an idca-format or idce-format DataFrame,
            with the caseid's and altid's in a two-level pandas MultiIndex.
        crack : bool, default True
            If True, the `dissolve_zero_variance` method is applied before
            repairing dtypes, to ensure that missing value are handled
            properly.
        altnames : Mapping, optional
            If given as a mapping, links alternative codes to names.
            An array or list of strings gives names for the alternatives,
            sorted in the same order as the codes.
        avail : str, default '_avail_'
            When the imported data is in idce format (i.e. sparse) then
            an availability indicator is computed and given this name. This
            argument has no effect if the data is already in idca format.
        fill_missing : scalar or Mapping, optional
            Fill values to use for missing values when imported data is
            in idce format (i.e. sparse).  Give a single value to use
            globally, or a mapping of {variable: value} or {dtype: value}.

        Returns
        -------
        Dataset

        See Also
        --------
        Dataset.dc.from_idce : Construct a Dataset from a sparse idca-format DataFrame.
        """
        return _DatasetConstruct.from_idca(
            df, crack=crack, altnames=altnames, avail=avail, fill_missing=fill_missing
        )

    @staticmethod
    def from_idce(
        df: pd.DataFrame,
        crack: bool = True,
        altnames: Mapping[int, str] | Sequence[str] = None,
        dim_name: str | None = None,
        alt_index: str = "alt_idx",
        case_index: str | None = None,
        case_pointer=None,
    ):
        """
        Construct a Dataset from a sparse idca-format DataFrame.

        Parameters
        ----------
        df : DataFrame
            The input data should be an idca-format or idce-format DataFrame,
            with the caseid's and altid's in a two-level pandas MultiIndex.
        crack : bool, default False
            If True, the `dissolve_zero_variance` method is applied before
            repairing dtypes, to ensure that missing value are handled
            properly.
        altnames : Mapping, optional
            If given as a mapping, links alternative codes to names.
            An array or list of strings gives names for the alternatives,
            sorted in the same order as the codes.
        dim_name : str, optional
            Name to apply to the sparse index dimension.
        alt_index : str, default 'alt_idx'
            Add the alt index (position) for each sparse data row as a
            coords array with this name.
        case_index : str, optional
            Add the case index (position) for each sparse data row as a
            coords array with this name. If not given, this array is not
            stored but it can still be reconstructed later from the case
            pointers.
        case_pointer : str, optional
            Use this name for the case_ptr dimension, overriding the
            default.

        Returns
        -------
        Dataset

        See Also
        --------
        Dataset.from_idca : Construct a dense Dataset from a idca-format DataFrame.
        """
        return _DatasetConstruct.from_idce(
            df,
            crack=crack,
            altnames=altnames,
            dim_name=dim_name,
            alt_index=alt_index,
            case_index=case_index,
            case_pointer=case_pointer,
        )

    def set_dtypes(self, dtypes, inplace=False, on_error="warn"):
        """
        Set the dtypes for the variables in this Dataset.

        Parameters
        ----------
        dtypes : Mapping or DataFrame
            Mapping of names to dtypes, or a DataFrame to infer such a
            mapping.
        inplace : bool, default False
            Whether to convert dtypes inplace.
        on_error : {'warn', 'raise', 'ignore'}
            What to do when a type conversion triggers an error.

        Returns
        -------
        Dataset
        """
        if isinstance(dtypes, pd.DataFrame):
            dtypes = dtypes.dtypes
        if inplace:
            obj = self._obj
        else:
            obj = self._obj.copy()
        for k in obj:
            if k not in dtypes:
                continue
            try:
                # originally, this used `obj[k] = obj[k].astype(dtypes[k])`
                # to convert the dtype of each variable in the Dataset.
                # However, this was found to be slow, as it checks all the
                # dimensions, indexes, and coordinates every time.  We know
                # that none of that changes so we don't need to do any checks.
                #
                # The `merge` method with `compat`, `join`, and `combine_attrs`
                # all set to "override" was also tried, but it was found to be
                # unsuccessful, as the dtype conversion did not work correctly.
                #
                # So we use the `data` attribute to directly set the data on
                # the variable we want to convert, which is much faster.  By
                # using `copy=False`, we avoid unnecessary copying of the data
                # when the dtype is already correct.
                obj[k].data = obj[k].astype(dtype=dtypes[k], copy=False).data

            except Exception as err:
                if on_error == "warn":
                    warnings.warn(
                        f"{err!r} on converting {k} to type {dtypes[k]}", stacklevel=2
                    )
                elif on_error == "raise":
                    raise
        return obj


@xr.register_dataset_accessor("icase")
class _DatasetCaseIslice(_GenericFlow):
    def __getitem__(self, item):
        if self.CASEID is not None:
            return self._obj.isel({self.CASEID: item})
        elif self.GROUPID is not None:
            return self._obj.isel({self.GROUPID: item})
        else:
            raise ValueError("neither CASEID nor GROUPID is defined")

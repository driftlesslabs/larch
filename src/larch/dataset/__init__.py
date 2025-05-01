from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
import sharrow as sh
import xarray as xr
from pandas.errors import UndefinedVariableError
from xarray.core import dtypes

from . import construct as construct
from . import flow as flow
from . import patch as patch
from .dim_names import ALTID as _ALTID
from .dim_names import ALTIDX as _ALTIDX
from .dim_names import CASEALT as _CASEALT
from .dim_names import CASEID as _CASEID
from .dim_names import CASEPTR as _CASEPTR
from .patch import register_dataarray_classmethod

# from .dim_names import GROUPID as _GROUPID
# from .dim_names import INGROUP as _INGROUP

try:
    from sharrow import DataArray as _sharrow_DataArray
    from sharrow import Dataset as _sharrow_Dataset
    from sharrow import DataTree as _sharrow_DataTree
    from sharrow.accessors import register_dataarray_method
except ImportError:
    warnings.warn("larch.dataset requires the sharrow library", stacklevel=2)

    class _noclass:
        pass

    _sharrow_Dataset = xr.Dataset
    _sharrow_DataArray = xr.DataArray
    _sharrow_DataTree = _noclass
    register_dataarray_method = lambda x: x


DataArray = _sharrow_DataArray


@register_dataarray_classmethod
def zeros(cls, *coords, dtype=np.float64, name=None, attrs=None):
    """
    Construct a dataset filled with zeros.

    Parameters
    ----------
    coords : Tuple[array-like]
        A sequence of coordinate vectors.  Ideally each should have a
        `name` attribute that names a dimension, otherwise placeholder
        names are used.
    dtype : dtype, default np.float64
        dtype of the new array. If omitted, it defaults to np.float64.
    name : str or None, optional
        Name of this array.
    attrs : dict_like or None, optional
        Attributes to assign to the new instance. By default, an empty
        attribute dictionary is initialized.

    Returns
    -------
    DataArray
    """
    dims = []
    shape = []
    coo = {}
    for n, c in enumerate(coords):
        i = getattr(c, "name", f"dim_{n}")
        dims.append(i)
        shape.append(len(c))
        coo[i] = c
    return cls(
        data=np.zeros(shape, dtype=dtype),
        coords=coo,
        dims=dims,
        name=name,
        attrs=attrs,
    )


@register_dataarray_classmethod
def ones(cls, *coords, dtype=np.float64, name=None, attrs=None):
    """
    Construct a dataset filled with ones.

    Parameters
    ----------
    coords : Tuple[array-like]
        A sequence of coordinate vectors.  Ideally each should have a
        `name` attribute that names a dimension, otherwise placeholder
        names are used.
    dtype : dtype, default np.float64
        dtype of the new array. If omitted, it defaults to np.float64.
    name : str or None, optional
        Name of this array.
    attrs : dict_like or None, optional
        Attributes to assign to the new instance. By default, an empty
        attribute dictionary is initialized.

    Returns
    -------
    DataArray
    """
    dims = []
    shape = []
    coo = {}
    for n, c in enumerate(coords):
        i = getattr(c, "name", f"dim_{n}")
        dims.append(i)
        shape.append(len(c))
        coo[i] = c
    return cls(
        data=np.ones(shape, dtype=dtype),
        coords=coo,
        dims=dims,
        name=name,
        attrs=attrs,
    )


@register_dataarray_classmethod
def from_zarr(cls, *args, name=None, **kwargs):
    dataset = xr.open_zarr(*args, **kwargs)
    if name is None:
        names = set(dataset.variables) - set(dataset.coords)
        if len(names) == 1:
            name = names.pop()
        else:
            raise ValueError("cannot infer name to load")
    return dataset[name]


@register_dataarray_method
def value_counts(self, index_name="index"):
    """
    Count the number of times each unique value appears in the array.

    Parameters
    ----------
    index_name : str, default 'index'
        Name of index dimension in result.

    Returns
    -------
    DataArray
    """
    values, freqs = np.unique(self, return_counts=True)
    return self.__class__(freqs, dims=index_name, coords={index_name: values})


Dataset = _sharrow_Dataset


class DataTree(_sharrow_DataTree):
    DatasetType = Dataset

    def __init__(
        self,
        graph=None,
        root_node_name=None,
        extra_funcs=(),
        extra_vars=None,
        cache_dir=None,
        relationships=(),
        force_digitization=False,
        **kwargs,
    ):
        super().__init__(
            graph=graph,
            root_node_name=root_node_name,
            extra_funcs=extra_funcs,
            extra_vars=extra_vars,
            cache_dir=cache_dir,
            relationships=relationships,
            force_digitization=force_digitization,
            **kwargs,
        )
        dim_order = []
        c = self.root_dataset.dc.CASEID
        if c is None and len(self.root_dataset.sizes) == 1:
            self.root_dataset.dc.CASEID = list(self.root_dataset.sizes.keys())[0]
        c = self.root_dataset.dc.CASEID
        if c is not None:
            dim_order.append(c)
        a = self.root_dataset.dc.ALTID
        if a is not None:
            dim_order.append(a)
        self.dim_order = tuple(dim_order)

    def idco_subtree(self):
        if "idcoVars" in self.subspaces:
            return self.subspaces["idcoVars"].dc.as_tree()
        return self.drop_dims(self.ALTID, ignore_missing_dims=True)

    @property
    def dc(self):
        return self

    @property
    def CASEID(self):
        """Str : The _caseid_ dimension of the root Dataset."""
        result = self.root_dataset.dc.CASEID
        if result is None:
            warnings.warn("no defined CASEID", stacklevel=2)
            return _CASEID
        return result

    @property
    def ALTID(self):
        """Str : The _altid_ dimension of the root Dataset."""
        result = self.root_dataset.dc.ALTID
        if result is None:
            warnings.warn("no defined ALTID", stacklevel=2)
            return _ALTID
        return result

    @property
    def CASEALT(self):
        """Str : The _casealt_ dimension of the root Dataset, if defined."""
        result = self.root_dataset.attrs.get(_CASEALT, None)
        return result

    @property
    def ALTIDX(self):
        """Str : The _alt_idx_ dimension of the root Dataset, if defined."""
        result = self.root_dataset.attrs.get(_ALTIDX, None)
        return result

    @property
    def CASEPTR(self):
        """Str : The _caseptr_ dimension of the root Dataset, if defined."""
        result = self.root_dataset.attrs.get(_CASEPTR, None)
        return result

    @property
    def n_cases(self):
        """Int : The size of the _caseid_ dimension of the root Dataset."""
        return self.root_dataset.sizes[self.CASEID]

    @property
    def n_alts(self):
        """Int : The size of the _altid_ dimension of the root Dataset."""
        return self.root_dataset.sizes[self.ALTID]

    def query_cases(self, query, parser="pandas", engine=None):
        """
        Return a new DataTree, with a query filter applied to the root Dataset.

        Parameters
        ----------
        query : str
            Python expressions to be evaluated against the data variables
            in the root dataset. The expressions will be evaluated using the pandas
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
        DataTree
            A new DataTree with the same contents as this DataTree, except each
            array of the root Dataset is indexed by the results of the query on
            the CASEID dimension.

        See Also
        --------
        Dataset.query_cases
        """
        obj = self.copy()
        try:
            obj.root_dataset = obj.root_dataset.dc.query_cases(
                query, parser=parser, engine=engine
            )
        except UndefinedVariableError:
            filter = self.idco_subtree().get_expr(
                query, allow_native=False, engine="sharrow", dtype="bool_"
            )
            obj.root_dataset = obj.root_dataset.dc.isel({self.CASEID: filter})
        return obj

    def slice_cases(self, *case_slice):
        if len(case_slice) != 1 or not isinstance(case_slice[0], slice):
            case_slice = slice(*case_slice)
        return self.replace_datasets(
            {self.root_node_name: self.root_dataset.isel({self.CASEID: case_slice})}
        )

    def caseids(self) -> pd.Index:
        """
        Access the caseids coordinates as an index.

        Returns
        -------
        pandas.Index
        """
        try:
            return self.root_dataset.indexes[self.CASEID]
        except KeyError:
            for _k, v in self.subspaces.items():
                if self.CASEID in v.indexes:
                    return v.indexes[self.CASEID]
            raise

    def altids(self) -> pd.Index:
        """
        Access the altids coordinates as an index.

        Returns
        -------
        pd.Index
        """
        try:
            return self.root_dataset.indexes[self.ALTID]
        except KeyError:
            for _k, v in self.subspaces.items():
                if self.ALTID in v.indexes:
                    return v.indexes[self.ALTID]
            raise

    def set_altnames(self, alt_names):
        """
        Set the alternative names for this DataTree.

        Parameters
        ----------
        altnames : Mapping or array-like
            A mapping of (integer) codes to names, or an array or names
            of the same length and order as the alternatives already
            defined in this Dataset.
        """
        self.root_dataset = self.root_dataset.dc.set_altnames(alt_names)

    def alts_mapping(self):
        return self.root_dataset.dc.alts_mapping

    def alts_name_to_id(self):
        return dict((j, i) for (i, j) in self.alts_mapping().items())

    def setup_flow(self, *args, **kwargs) -> sh.Flow:
        """
        Set up a new Flow for analysis using the structure of this DataTree.

        Parameters
        ----------
        definition_spec : dict[str,str]
            Gives the names and expressions that define the variables to
            create in this new `Flow`.
        cache_dir : Path-like, optional
            A location to write out generated python and numba code. If not
            provided, a unique temporary directory is created.
        name : str, optional
            The name of this Flow used for writing out cached files. If not
            provided, a unique name is generated. If `cache_dir` is given,
            be sure to avoid name conflicts with other flow's in the same
            directory.
        dtype : str, default "float32"
            The name of the numpy dtype that will be used for the output.
        boundscheck : bool, default False
            If True, boundscheck enables bounds checking for array indices, and
            out of bounds accesses will raise IndexError. The default is to not
            do bounds checking, which is faster but can produce garbage results
            or segfaults if there are problems, so try turning this on for
            debugging if you are getting unexplained errors or crashes.
        error_model : {'numpy', 'python'}, default 'numpy'
            The error_model option controls the divide-by-zero behavior. Setting
            it to ‘python’ causes divide-by-zero to raise exception like
            CPython. Setting it to ‘numpy’ causes divide-by-zero to set the
            result to +/-inf or nan.
        nopython : bool, default True
            Compile using numba's `nopython` mode.  Provided for debugging only,
            as there's little point in turning this off for production code, as
            all the speed benefits of sharrow will be lost.
        fastmath : bool, default True
            If true, fastmath enables the use of "fast" floating point transforms,
            which can improve performance but can result in tiny distortions in
            results.  See numba docs for details.
        parallel : bool, default True
            Enable or disable parallel computation for certain functions.
        readme : str, optional
            A string to inject as a comment at the top of the flow Python file.
        flow_library : Mapping[str,Flow], optional
            An in-memory cache of precompiled Flow objects.  Using this can result
            in performance improvements when repeatedly using the same definitions.
        extra_hash_data : Tuple[Hashable], optional
            Additional data used for generating the flow hash.  Useful to prevent
            conflicts when using a flow_library with multiple similar flows.
        write_hash_audit : bool, default True
            Writes a hash audit log into a comment in the flow Python file, for
            debugging purposes.
        hashing_level : int, default 1
            Level of detail to write into flow hashes.  Increase detail to avoid
            hash conflicts for similar flows.  Level 2 adds information about
            names used in expressions and digital encodings to the flow hash,
            which prevents conflicts but requires more pre-computation to generate
            the hash.
        dim_exclude : Collection[str], optional
            Exclude these root dataset dimensions from this flow.

        Returns
        -------
        Flow
        """
        if "dim_exclude" not in kwargs:
            if "_exclude_dims_" in self.root_dataset.attrs:
                kwargs["dim_exclude"] = self.root_dataset.attrs["_exclude_dims_"]
        try:
            return super().setup_flow(*args, **kwargs)
        except ValueError as err:
            regex = re.match("^unable to rewrite (.*) to itself$", str(err))
            if regex:
                raise ValueError(
                    f"Setup failed for variable {regex.group(1)}.  Check the expression "
                    f"and the names of the variables in the dataset."
                ) from err
            else:
                raise err


def merge(
    objects,
    compat="no_conflicts",
    join="outer",
    fill_value=dtypes.NA,
    combine_attrs="override",
    *,
    caseid=None,
    alts=None,
):
    """
    Merge any number of xarray objects into a single larch.Dataset as variables.

    Parameters
    ----------
    objects : iterable of Dataset or iterable of DataArray or iterable of dict-like
        Merge together all variables from these objects. If any of them are
        DataArray objects, they must have a name.

    compat : {"identical", "equals", "broadcast_equals", "no_conflicts", "override"}, optional
        String indicating how to compare variables of the same name for
        potential conflicts:
        - "broadcast_equals": all values must be equal when variables are
          broadcast against each other to ensure common dimensions.
        - "equals": all values and dimensions must be the same.
        - "identical": all values, dimensions and attributes must be the
          same.
        - "no_conflicts": only values which are not null in both datasets
          must be equal. The returned dataset then contains the combination
          of all non-null values.
        - "override": skip comparing and pick variable from first dataset

    join : {"outer", "inner", "left", "right", "exact"}, optional
        String indicating how to combine differing indexes in objects.
        - "outer": use the union of object indexes
        - "inner": use the intersection of object indexes
        - "left": use indexes from the first object with each dimension
        - "right": use indexes from the last object with each dimension
        - "exact": instead of aligning, raise `ValueError` when indexes to be
          aligned are not equal
        - "override": if indexes are of same size, rewrite indexes to be
          those of the first object with that dimension. Indexes for the same
          dimension must have the same size in all objects.

    fill_value : scalar or dict-like, optional
        Value to use for newly missing values. If a dict-like, maps
        variable names to fill values. Use a data array's name to
        refer to its values.

    combine_attrs : {"drop", "identical", "no_conflicts", "drop_conflicts", \
                    "override"} or callable, default: "override"
        A callable or a string indicating how to combine attrs of the objects being
        merged:
        - "drop": empty attrs on returned Dataset.
        - "identical": all attrs must be the same on every object.
        - "no_conflicts": attrs from all objects are combined, any that have
          the same name must also have the same value.
        - "drop_conflicts": attrs from all objects are combined, any that have
          the same name but different values are dropped.
        - "override": skip comparing and copy attrs from the first dataset to
          the result.
        If a callable, it must expect a sequence of ``attrs`` dicts and a context object
        as its only parameters.

    caseid : str, optional, keyword only
        This named dimension will be marked as the '_caseid_' dimension.

    alts : str or Mapping or array-like, keyword only
        If given as a str, this named dimension will be marked as the
        '_altid_' dimension.  Otherwise, give a Mapping that defines
        alternative names and (integer) codes or an array of codes.

    Returns
    -------
    Dataset
        Dataset with combined variables from each object.
    """
    return Dataset.construct(
        xr.merge(
            objects,
            compat="no_conflicts",
            join="outer",
            fill_value=dtypes.NA,
            combine_attrs="override",
        ),
        caseid=caseid,
        alts=alts,
    )


# @nb.njit
# def ce_dissolve_zero_variance(ce_data, ce_caseptr):
#     """
#
#     Parameters
#     ----------
#     ce_data : array-like, shape [n_casealts] one-dim only
#     ce_altidx
#     ce_caseptr
#     n_alts
#
#     Returns
#     -------
#     out : ndarray
#     flag : int
#         1 if variance was detected, 0 if no variance was found and
#         the `out` array is valid.
#     """
#     failed = 0
#     if ce_caseptr.ndim == 2:
#         ce_caseptr1 = ce_caseptr[:,-1]
#     else:
#         ce_caseptr1 = ce_caseptr[1:]
#     shape = (ce_caseptr1.shape[0], )
#     out = np.zeros(shape, dtype=ce_data.dtype)
#     c = 0
#     out[0] = ce_data[0]
#     for row in range(ce_data.shape[0]):
#         if row == ce_caseptr1[c]:
#             c += 1
#             out[c] = ce_data[row]
#         else:
#             if out[c] != ce_data[row]:
#                 failed = 1
#                 break
#     return out, failed


# @nb.njit
# def case_ptr_to_indexes(n_casealts, case_ptrs):
#     case_index = np.zeros(n_casealts, dtype=np.int64)
#     for c in range(case_ptrs.shape[0]-1):
#         case_index[case_ptrs[c]:case_ptrs[c + 1]] = c
#     return case_index

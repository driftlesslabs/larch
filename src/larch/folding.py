from __future__ import annotations

import numpy as np
import xarray as xr
from xarray import Dataset

_GROUPID = "_groupid_"
_INGROUP = "_ingroup_"


def random_groups():
    pass


def _group_breaks(groupid):
    group_breaks = np.append(
        np.insert(np.where(np.diff(groupid))[0] + 1, 0, 0), len(groupid)
    )
    lens = np.diff(group_breaks)
    pad_len = np.max(lens) - lens
    where_to_pad = np.repeat(group_breaks[1:], pad_len)
    ids = groupid[group_breaks[:-1]]
    return lens, where_to_pad, ids


def fold(
    arr,
    groupid,
    fill_value=0,
    group_dim="groupid",
    ingroup_dim="ingroup",
    lens=None,
    where_to_pad=None,
    gids=None,
):
    try:
        arr_GROUPID = arr.attrs.get(_GROUPID, None)
    except AttributeError:
        pass
    else:
        if arr_GROUPID == group_dim:
            # assume already folded
            # TODO check
            return arr
    if arr.shape[0] < groupid.size:
        # assume already folded
        # TODO check
        return arr
    if lens is None or where_to_pad is None:
        lens, where_to_pad, gids = _group_breaks(groupid)
        # lens = np.diff(group_breaks)
        # pad_len = np.max(lens) - lens
        # where_to_pad = np.repeat(group_breaks[1:], pad_len)
    padding_value = np.full(
        arr.shape[1:],
        fill_value,
    )
    reshape_to = [-1, np.max(lens), *arr.shape[1:]]
    bigger = np.insert(np.asarray(arr), where_to_pad, padding_value, axis=0)
    try:
        padded_arr = bigger.reshape(reshape_to)
    except ValueError:
        if bigger.size == 0:
            padded_arr = bigger.reshape(
                [int(bigger.shape[0] / np.max(lens)), np.max(lens), *arr.shape[1:]]
            )
        else:
            raise
    if isinstance(arr, xr.DataArray):
        return arr.__class__(
            padded_arr,
            dims=[group_dim, ingroup_dim, *arr.dims[1:]],
            coords={group_dim: np.asarray(gids)} if gids is not None else {},
            name=arr.name,
        )
    return padded_arr


def fold_dataset(
    d,
    groupid="groupid",
    group_dim="groupid",
    ingroup_dim="ingroup",
    vars=("ca", "co", "ch", "av", "caseid"),
    collapse_zero_variance=False,
):
    try:
        if group_dim in d.dims and ingroup_dim in d.dims:
            return d  # already folded!
    except AttributeError:
        raise
    if isinstance(groupid, str):
        if groupid not in d:
            return d
        g = d[groupid]
    else:
        g = groupid
    out = Dataset()
    lens, where_to_pad, gids = _group_breaks(g)
    if vars is None:
        vars = [i for i in d.variables if i not in (group_dim, ingroup_dim)]
    for i in vars:
        if i in d:
            temp = fold(
                d[i],
                g,
                group_dim=group_dim,
                ingroup_dim=ingroup_dim,
                lens=lens,
                where_to_pad=where_to_pad,
                gids=gids,
            )
            out = out.assign({i: temp})
    for c in d.coords:
        if d.dc.CASEID not in d[c].dims:
            out.coords[c] = d.coords[c]
    out.dc.ALTID = d.dc.ALTID
    try:
        out.dc.GROUPID = group_dim
        out.dc.INGROUP = ingroup_dim
    except Exception:
        pass
    return out


# def folder(cls, ds, fold_on, crack=True, avail="_avail_", fill_missing=None):
#     """
#     Construct a Dataset from an idca-format DataFrame.
#
#     This method loads the data as dense arrays.
#
#     Parameters
#     ----------
#     ds : Dataset
#         The input data should be an idca-format or idce-format DataFrame,
#         with the caseid's and altid's in a two-level pandas MultiIndex.
#     crack : bool, default True
#         If True, the `dissolve_zero_variance` method is applied before
#         repairing dtypes, to ensure that missing value are handled
#         properly.
#     avail : str, default '_avail_'
#         When the imported data is sparse then
#         an availability indicator is computed and given this name.
#     fill_missing : scalar or Mapping, optional
#         Fill values to use for missing values when imported data is
#         sparse.  Give a single value to use
#         globally, or a mapping of {variable: value} or {dtype: value}.
#
#     Returns
#     -------
#     Dataset
#
#     """
#     # if df.index.nlevels != 2:
#     #     raise ValueError("source idca dataframe must have a two "
#     #                      "level MultiIndex giving case and alt id's")
#     # caseidname, altidname = df.index.names
#
#     # check altids are integers, if they are not then fix it
#     # if df.index.levels[1].dtype.kind != 'i':
#     #     if altnames is None:
#     #         altnames = df.index.levels[1]
#     #         df.index = df.index.set_levels(np.arange(1, len(altnames) + 1), level=1)
#     #     else:
#     #         new_index = df.index.get_level_values(1).astype(pd.CategoricalDtype(altnames))
#     #         df.index = df.index.set_codes(
#     #             new_index.codes, level=1
#     #         ).set_levels(np.arange(1, len(altnames) + 1), level=1)
#
#     ds = cls()(df, caseid=caseidname, alts=altidname)
#     if crack:
#         ds = ds.dc.dissolve_zero_variance()
#     ds = ds.dc.set_dtypes(df)
#     if altnames is not None:
#         ds = ds.dc.set_altnames(altnames)
#     if avail not in ds and len(df) < ds.dc.n_cases * ds.dc.n_alts:
#         av = (
#             xr.DataArray.from_series(pd.Series(1, index=df.index))
#             .fillna(0)
#             .astype(np.int8)
#         )
#         ds[avail] = av
#         if fill_missing is not None:
#             if isinstance(fill_missing, Mapping):
#                 for k, i in ds.items():
#                     if ds.dc.ALTID not in i.dims:
#                         continue
#                     if k not in fill_missing and i.dtype not in fill_missing:
#                         continue
#                     filler = fill_missing.get(k, fill_missing[i.dtype])
#                     ds[k] = i.where(ds["_avail_"] != 0, filler)
#             else:
#                 for k, i in ds.items():
#                     if ds.dc.ALTID not in i.dims:
#                         continue
#                     ds[k] = i.where(ds["_avail_"] != 0, fill_missing)
#     return ds


def dissolve_zero_variance(self, dim, mask=None, inplace=False):
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
    Dataset
    """
    if inplace:
        obj = self
    else:
        obj = self.copy()
    for k in obj.variables:
        if obj[k].dtype.kind in {"U", "S", "O"}:
            continue
        if dim in obj[k].dims:
            try:
                if mask:
                    dissolve = obj[k].where(self[mask]).std(dim=dim).max() < 1e-10
                else:
                    dissolve = obj[k].std(dim=dim).max() < 1e-10
            except TypeError:
                pass
            else:
                if dissolve:
                    if mask:
                        obj[k] = (
                            obj[k].where(self[mask]).min(dim=dim).astype(obj[k].dtype)
                        )
                    else:
                        obj[k] = obj[k].min(dim=dim)
    return obj

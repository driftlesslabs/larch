import numpy as np
import xarray as xr
from xarray import DataArray, Dataset
import warnings

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

def fold(arr, groupid, fill_value=0, group_dim='groupid', ingroup_dim='ingroup', lens=None, where_to_pad=None, gids=None):
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
    padding_value = np.full(arr.shape[1:], fill_value,)
    reshape_to = [-1, np.max(lens), *arr.shape[1:]]
    bigger = np.insert(np.asarray(arr), where_to_pad, padding_value, axis=0)
    padded_arr = bigger.reshape(
        reshape_to
    )
    if isinstance(arr, xr.DataArray):
        return arr.__class__(
            padded_arr,
            dims=[group_dim, ingroup_dim, *arr.dims[1:]],
            coords={group_dim: np.asarray(gids)} if gids is not None else {},
            name=arr.name,
        )
    return padded_arr


def fold_dataset(d, groupid='groupid', group_dim='groupid', ingroup_dim='ingroup', vars=('ca', 'co', 'ch', 'av', 'caseid')):
    if group_dim in d.dims and ingroup_dim in d.dims:
        return d # already folded!
    if isinstance(groupid, str):
        if groupid not in d:
            return d
        g = d[groupid]
    else:
        g = groupid
    out = Dataset()
    lens, where_to_pad, gids = _group_breaks(g)
    if vars is None:
        vars = d.variables
    for i in vars:
        if i in d:
            temp = fold(d[i], g, group_dim=group_dim, ingroup_dim=ingroup_dim, lens=lens, where_to_pad=where_to_pad, gids=gids)
            out = out.assign({i:temp})
    for c in d.coords:
        if d.dc.CASEID not in d[c].dims:
            out.coords[c] = d.coords[c]
    out.dc.ALTID = d.dc.ALTID
    out.dc.GROUPID = group_dim
    out.dc.INGROUP = ingroup_dim
    return out

import numpy as np
import xarray as xr
from larch.numba import DataArray, Dataset

def random_groups():
    pass


def fold(arr, groupid, fill_value=0):
    if arr.shape[0] < groupid.size:
        # assume already folded
        # TODO check
        return arr
    group_breaks = np.append(
        np.insert(np.where(np.diff(groupid))[0] + 1, 0, 0), len(groupid)
    )
    lens = np.diff(group_breaks)
    pad_len = np.max(lens) - lens
    where_to_pad = np.repeat(group_breaks[1:], pad_len)
    padding_value = np.full(arr.shape[1:], fill_value,)
    reshape_to = [-1, np.max(lens), *arr.shape[1:]]
    bigger = np.insert(np.asarray(arr), where_to_pad, padding_value, axis=0)
    padded_arr = bigger.reshape(
        reshape_to
    )
    if isinstance(arr, xr.DataArray):
        return arr.__class__(
            padded_arr,
            dims=['groupid', 'ingroupid', *arr.dims[1:]],
        )
    return padded_arr


def fold_dataset(d, groupid='groupid'):
    if isinstance(groupid, str):
        if groupid not in d.coords:
            return d
        g = d.coords[groupid]
    else:
        g = groupid
    out = Dataset()
    for i in ['ca', 'co', 'ch', 'av', 'caseid']:
        if i in d:
            out[i] = fold(d[i], g)
    for c in d.coords:
        if d.CASEID not in d[c].dims:
            out.coords[c] = d.coords[c]
    out.ALTID = d.ALTID
    return out

from __future__ import annotations

import logging
from typing import NamedTuple

import numba as nb
import numpy as np

from ..dataset.dim_names import ALTID as _ALTID
from ..dataset.dim_names import CASEALT as _CASEALT
from ..dataset.dim_names import CASEID as _CASEID


class _case_slice:
    def __get__(self, obj, objtype=None):
        self.parent = obj
        return self

    def __getitem__(self, idx):
        kwds = {}
        for k in self.parent._fields:
            attribute = getattr(self.parent, k)
            if attribute is not None and (len(k) == 2 or k == "ce_caseptr"):
                kwds[k] = attribute[idx]
            else:
                kwds[k] = attribute
        return type(self.parent)(**kwds)


class DataArrays(NamedTuple):
    ch: np.ndarray
    av: np.ndarray
    wt: np.ndarray
    co: np.ndarray
    ca: np.ndarray
    ce_data: np.ndarray
    ce_altidx: np.ndarray
    ce_caseptr: np.ndarray

    cs = _case_slice()

    # @property
    # def alternatives(self):
    #     if self.alt_codes is not None:
    #         if self.alt_names is not None:
    #             return dict(zip(self.alt_codes, self.alt_names))
    #         else:
    #             return {i:str(i) for i in self.alt_codes}
    #     else:
    #         raise ValueError("alt_codes not defined")


def to_dataset(dataframes):
    caseindex_name = _CASEID
    altindex_name = _ALTID
    from xarray import DataArray

    from ..dataset import Dataset

    coords = {
        caseindex_name: dataframes.caseindex.values,
        altindex_name: dataframes.alternative_codes(),
    }
    ds = Dataset(coords=coords)
    if dataframes.data_co is not None:
        caseindex_name = dataframes.data_co.index.name
        ds.update(Dataset.from_dataframe(dataframes.data_co))
    if dataframes.data_ca is not None:
        caseindex_name = dataframes.data_ca.index.names[0]
        altindex_name = dataframes.data_ca.index.names[1]
        ds.update(Dataset.from_dataframe(dataframes.data_ca))
    alt_names = dataframes.alternative_names()
    if alt_names:
        ds.coords["alt_names"] = DataArray(alt_names, dims=(altindex_name,))
    return ds


def prepare_data(
    datasource,
    request,
    float_dtype=None,
    cache_dir=None,
    flows=None,
    make_unused_flows=False,
):
    """
    Load data from a DataTree into a computationally-formatted Dataset.

    Parameters
    ----------
    datasource : DataTree or Dataset
    request : Mapping or Model
    float_dtype : dtype, optional
    cache_dir : Path-like, optional
        Directory to cache sharrow flows.
    flows : dict, optional
        Collection of previously prepared flows.

    Returns
    -------
    model_dataset : Dataset
        Ready for computation.
    flows : dict
    """
    log = logging.getLogger("Larch")
    from ..dataset import DataArray, Dataset, DataTree

    if float_dtype is None:
        float_dtype = np.float64
    log.debug(f"building dataset from datashare coords: {datasource.coords}")
    model_dataset = Dataset(
        coords=datasource.coords,
    )
    try:
        if "idcoVars" in datasource.subspaces:
            model_dataset.coords.update(datasource.subspaces["idcoVars"].coords)
    except AttributeError:
        pass
    model_dataset.dc.CASEID = datasource.dc.CASEID
    model_dataset.dc.ALTID = datasource.dc.ALTID

    from .numbamodel import NumbaModel  # avoid circular import

    if isinstance(request, NumbaModel):
        alts = request.graph.elemental_names()
        alt_dim = model_dataset.dc.ALTID or _ALTID
        if model_dataset.dc.ALTID not in model_dataset.coords:
            model_dataset.coords[alt_dim] = DataArray(
                list(alts.keys()), dims=(alt_dim,)
            )
        if "alt_names" not in model_dataset.coords:
            model_dataset.coords["alt_names"] = DataArray(
                list(alts.values()), dims=(alt_dim,)
            )
        request = request.required_data()

    if flows is None:
        flows = {}

    if isinstance(datasource, DataTree):
        log.debug("adopting existing DataTree")
        if not datasource.relationships_are_digitized:
            datasource.digitize_relationships(inplace=True)
        datatree = datasource
        datatree_co = datatree.idco_subtree()
    elif isinstance(datasource, Dataset):
        datatree = datasource.dc.as_tree()
        if not datatree.relationships_are_digitized:
            datatree.digitize_relationships(inplace=True)
        datatree_co = datatree.idco_subtree()
    else:
        log.debug("initializing new DataTree")
        datatree = DataTree(main=datasource)
        datatree.digitize_relationships(inplace=True)
        datatree_co = datatree.idco_subtree()

    if "co" in request:
        log.debug(f"requested co data: {request['co']}")
        model_dataset, flows["co"] = _prep_co(
            model_dataset,
            datatree_co,
            request["co"],
            tag="co",
            dtype=float_dtype,
            cache_dir=cache_dir,
            flow=flows.get("co"),
        )
    if "ca" in request:
        log.debug(f"requested ca data: {request['ca']}")
        casealt_dim = datatree.root_dataset.attrs.get(_CASEALT)
        if casealt_dim is None:
            model_dataset, flows["ca"] = _prep_ca(
                model_dataset,
                datatree,
                request["ca"],
                tag="ca",
                dtype=float_dtype,
                cache_dir=cache_dir,
                flow=flows.get("ca"),
            )
        else:
            model_dataset, flows["ce"] = _prep_ce(
                model_dataset,
                datatree,
                request["ca"],
                dtype=float_dtype,
                cache_dir=cache_dir,
                flow=flows.get("ce"),
            )
    if "choice_ca" in request:
        log.debug(f"requested choice_ca data: {request['choice_ca']}")
        casealt_dim = datatree.root_dataset.attrs.get(_CASEALT)
        if casealt_dim is None:
            log.debug("  loading choice_ca data from idca")
            model_dataset, flows["choice_ca"] = _prep_ca(
                model_dataset,
                datatree,
                request["choice_ca"],
                tag="ch",
                preserve_vars=False,
                dtype=float_dtype,
                cache_dir=cache_dir,
                flow=flows.get("choice_ca"),
                force_flow=make_unused_flows,
            )
        else:
            log.debug("  loading choice_ca data from idce")
            model_dataset, flows["choice_ce"] = _prep_ce(
                model_dataset,
                datatree,
                request["choice_ca"],
                v_tag="choice_ca",
                s_tag="choice_ce",
                preserve_vars=False,
                dtype=float_dtype,
                cache_dir=cache_dir,
                flow=flows.get("choice_ce"),
                attach_indexes=False,
                # force_flow=make_unused_flows,
            )
            da_ch = DataArray(
                ce_to_dense(
                    model_dataset["choice_ce_data"].values,
                    model_dataset[model_dataset.dc.ALTIDX].values,
                    model_dataset[model_dataset.dc.CASEPTR].values,
                    datatree.n_alts,
                ),
                dims=[datatree.CASEID, datatree.ALTID],
                coords={
                    datatree.CASEID: model_dataset.coords[datatree.CASEID],
                    datatree.ALTID: model_dataset.coords[datatree.ALTID],
                },
                name="ch",
            )
            model_dataset = model_dataset.drop_vars(["choice_ce_data"])
            model_dataset["ch"] = da_ch
    if "choice_co_code" in request:
        log.debug(f"requested choice_co_code data: {request['choice_co_code']}")
        if isinstance(datasource, DataTree):
            choicecodes = datasource.get(
                request["choice_co_code"], broadcast=False, coords=False
            )
        else:
            choicecodes = datasource.get(request["choice_co_code"])
        da_ch = DataArray(
            np.zeros(
                shape=(
                    model_dataset.sizes[datatree.CASEID],
                    model_dataset.sizes[datatree.ALTID],
                ),
                dtype=float_dtype,
            ),
            dims=[datatree.CASEID, datatree.ALTID],
            coords={
                datatree.CASEID: model_dataset.coords[datatree.CASEID],
                datatree.ALTID: model_dataset.coords[datatree.ALTID],
            },
            name="ch",
        )
        for i, a in enumerate(model_dataset.coords[datatree.ALTID]):
            if isinstance(a, DataArray):
                a = a.item()
            da_ch[:, i] = np.asarray(choicecodes == a)
        model_dataset = model_dataset.merge(da_ch)
        if make_unused_flows:
            from .numba_stream import OneHotStreamer

            flows["choice_co_code"] = OneHotStreamer(
                request["choice_co_code"], datatree.root_dataset.coords[datatree.ALTID]
            )
    if "choice_co" in request:
        log.debug(f"requested choice_co_vars data: {request['choice_co']}")
        da_ch = DataArray(
            np.zeros(
                shape=(
                    model_dataset.sizes[datatree.CASEID],
                    model_dataset.sizes[datatree.ALTID],
                ),
                dtype=float_dtype,
            ),
            dims=[datatree.CASEID, datatree.ALTID],
            coords={
                datatree.CASEID: model_dataset.coords[datatree.CASEID],
                datatree.ALTID: model_dataset.coords[datatree.ALTID],
            },
            name="ch",
        )
        for i, a in enumerate(model_dataset.dc.alts_mapping):
            choice_expr = request["choice_co"][a]
            da_ch[:, i] = datatree_co.dc.get_expr(choice_expr).values
        model_dataset = model_dataset.merge(da_ch)
    if "choice_any" in request:
        log.debug(f"requested choice_any data: {request['choice_any']}")
        raise NotImplementedError("choice_any")

    if "weight_co" in request:
        log.debug(f"requested weight_co data: {request['weight_co']}")
        model_dataset, flows["weight_co"] = _prep_co(
            model_dataset,
            datatree_co,
            [request["weight_co"]],
            tag="wt",
            preserve_vars=make_unused_flows,
            dtype=float_dtype,
            cache_dir=cache_dir,
            flow=flows.get("weight_co"),
        )

    if "group_co" in request:
        log.debug(f"requested group_co data: {request['group_co']}")
        model_dataset, flows["group_co"] = _prep_co(
            model_dataset,
            datatree_co,
            [request["group_co"]],
            tag="group",
            preserve_vars=make_unused_flows,
            dtype=np.int64,
            cache_dir=cache_dir,
            flow=flows.get("group_co"),
        )

    if "avail_ca" in request:
        log.debug(f"requested avail_ca data: {request['avail_ca']}")
        casealt_dim = datatree.root_dataset.attrs.get(_CASEALT)
        if casealt_dim is None:
            model_dataset, flows["avail_ca"] = _prep_ca(
                model_dataset,
                datatree,
                request["avail_ca"],
                tag="av",
                preserve_vars=False,
                dtype=np.int8,
                cache_dir=cache_dir,
                flow=flows.get("avail_ca"),
                force_flow=make_unused_flows,
            )
        else:
            if (
                request["avail_ca"] in {"1", "True", "1.0"}
                and model_dataset.dc.CASEPTR is not None
                and model_dataset.dc.ALTIDX is not None
            ):
                da_av = DataArray(
                    ce_bool_to_dense(
                        model_dataset[model_dataset.dc.ALTIDX].values,
                        model_dataset[model_dataset.dc.CASEPTR].values,
                        datatree.n_alts,
                    ),
                    dims=[datatree.CASEID, datatree.ALTID],
                    coords={
                        datatree.CASEID: model_dataset.coords[datatree.CASEID],
                        datatree.ALTID: model_dataset.coords[datatree.ALTID],
                    },
                    name="av",
                )
            else:
                model_dataset, flows["avail_ce"] = _prep_ce(
                    model_dataset,
                    datatree,
                    request["avail_ca"],
                    v_tag="avail_ca",
                    s_tag="avail_ce",
                    preserve_vars=False,
                    dtype=np.int8,
                    cache_dir=cache_dir,
                    flow=flows.get("avail_ce"),
                    attach_indexes=False,
                )
                da_av = DataArray(
                    ce_to_dense(
                        model_dataset["avail_ce_data"].values,
                        model_dataset[model_dataset.dc.ALTIDX].values,
                        model_dataset[model_dataset.dc.CASEPTR].values,
                        datatree.n_alts,
                    ),
                    dims=[datatree.CASEID, datatree.ALTID],
                    coords={
                        datatree.CASEID: model_dataset.coords[datatree.CASEID],
                        datatree.ALTID: model_dataset.coords[datatree.ALTID],
                    },
                    name="av",
                )
                model_dataset = model_dataset.drop_vars(["avail_ce_data"])
            model_dataset["av"] = da_av
    if "avail_co" in request:
        log.debug(f"requested avail_co data: {request['avail_co']}")
        av_co_expressions = {
            a: request["avail_co"].get(a, "0")
            for a in model_dataset.coords[datatree.ALTID].values
        }
        model_dataset, flows["avail_co"] = _prep_co(
            model_dataset,
            datatree_co,
            av_co_expressions,
            tag="av",
            preserve_vars=make_unused_flows,
            dtype=np.int8,
            dim_name=datatree.ALTID,
            cache_dir=cache_dir,
            flow=flows.get("avail_co"),
        )
    if "avail_any" in request:
        log.debug(f"requested avail_any data: {request['avail_any']}")
        model_dataset = model_dataset.assign(
            {
                "av": DataArray.ones(
                    model_dataset.dc.caseids(),
                    model_dataset.dc.altids(),
                    dtype=np.int8,
                ),
            }
        )

    return model_dataset, flows


def flownamer(tag, definition_spec, extra_hash_features=()):
    import base64
    import hashlib

    defs_hash = hashlib.md5()
    defs_hash.update(str(tag).encode("utf8"))
    for k, v in definition_spec.items():
        defs_hash.update(str(k).encode("utf8"))
        defs_hash.update(str(v).encode("utf8"))
    for k in extra_hash_features:
        defs_hash.update(str(k).encode("utf8"))
    return "pipeline_" + (base64.b32encode(defs_hash.digest())).decode().replace(
        "=", ""
    )


def _prep_ca(
    model_dataset,
    shared_data_ca,
    vars_ca,
    tag="ca",
    preserve_vars=True,
    dtype=None,
    cache_dir=None,
    flow=None,
    force_flow=False,
    use_array_maker=True,
):
    from ..dataset import DataArray, DataTree

    assert isinstance(shared_data_ca, DataTree)
    if isinstance(vars_ca, str):
        if (
            not preserve_vars
            and vars_ca in shared_data_ca.root_dataset
            and not force_flow
        ):
            proposal = shared_data_ca.root_dataset[vars_ca]
            if (
                shared_data_ca.CASEID in proposal.sizes
                and shared_data_ca.ALTID in proposal.sizes
            ):
                proposal = proposal.drop_vars(list(proposal.coords)).rename(tag)
                # check if proposal.dtype matches the requested dtype (if any)
                if dtype is not None and proposal.dtype != dtype:
                    # if the requested dtype is an integer type but the native
                    # dtype is float, fill any missing values with zeros
                    if np.issubdtype(dtype, np.integer) and np.issubdtype(
                        proposal.dtype, np.floating
                    ):
                        proposal = proposal.fillna(0)
                    proposal = proposal.astype(dtype)
                return model_dataset.merge(proposal), vars_ca
    if isinstance(vars_ca, str):
        vars_ca = {vars_ca: vars_ca}
    if not isinstance(vars_ca, dict):
        vars_ca = {i: i for i in vars_ca}
    flowname = flownamer(tag, vars_ca, shared_data_ca._hash_features())
    if isinstance(flow, str):
        raise ValueError("expected flow {flow!r}")
    if flow is None or flowname != flow.name:
        flow = shared_data_ca.setup_flow(vars_ca, cache_dir=cache_dir, name=flowname)
    try:
        arr = flow.load(
            shared_data_ca,
            dtype=dtype,
            use_array_maker=use_array_maker,
        )
    except NameError:
        # the original resolution of the flow failed, try again with a fresh flow
        flow = shared_data_ca.setup_flow(vars_ca, cache_dir=cache_dir, hashing_level=2)
        arr = flow.load(
            shared_data_ca,
            dtype=dtype,
            use_array_maker=use_array_maker,
        )

    caseid_dim = shared_data_ca.CASEID
    altid_dim = shared_data_ca.ALTID
    if preserve_vars or len(vars_ca) > 1:
        arr = arr.reshape(
            model_dataset.sizes.get(caseid_dim),
            model_dataset.sizes.get(altid_dim),
            -1,
        )
        da = DataArray(
            arr,
            dims=[caseid_dim, altid_dim, f"var_{tag}"],
            coords={
                caseid_dim: model_dataset.coords[caseid_dim],
                altid_dim: model_dataset.coords[altid_dim],
                f"var_{tag}": list(vars_ca.keys()),
            },
            name=tag,
        )
    else:
        arr = arr.reshape(
            model_dataset.sizes.get(caseid_dim),
            model_dataset.sizes.get(altid_dim),
        )
        da = DataArray(
            arr,
            dims=[caseid_dim, altid_dim],
            coords={
                caseid_dim: model_dataset.coords[caseid_dim],
                altid_dim: model_dataset.coords[altid_dim],
            },
            name=tag,
        )
    return model_dataset.merge(da), flow


def _prep_ce(
    model_dataset,
    datatree,
    vars_ca,
    v_tag="ca",
    s_tag="ce",
    preserve_vars=True,
    dtype=None,
    cache_dir=None,
    flow=None,
    attach_indexes=True,
):
    from ..dataset import DataArray, DataTree

    assert isinstance(datatree, DataTree)
    if isinstance(vars_ca, str):
        if not preserve_vars and vars_ca in datatree.root_dataset:
            proposal = datatree.root_dataset[vars_ca]
            if datatree.CASEALT in proposal.sizes:
                proposal = proposal.drop_vars(list(proposal.coords)).rename(
                    f"{s_tag}_data"
                )
                return model_dataset.merge(proposal), flow
    if isinstance(vars_ca, str):
        vars_ca = {vars_ca: vars_ca}
    if not isinstance(vars_ca, dict):
        vars_ca = {i: i for i in vars_ca}
    flowname = flownamer(s_tag, vars_ca, datatree._hash_features())
    if flow is None or flowname != flow.name:
        flow = datatree.setup_flow(vars_ca, cache_dir=cache_dir, name=flowname)
    arr = flow.load(
        datatree,
        dtype=dtype,
    )
    casealt_dim = datatree.CASEALT
    if preserve_vars or len(vars_ca) > 1:
        arr = arr.reshape(
            model_dataset.sizes.get(casealt_dim),
            -1,
        )
        da = DataArray(
            arr,
            dims=[casealt_dim, f"var_{v_tag}"],
            coords={
                casealt_dim: model_dataset.coords[casealt_dim],
                f"var_{v_tag}": list(vars_ca.keys()),
            },
            name=f"{s_tag}_data",
        )
    else:
        arr = arr.reshape(-1)
        da = DataArray(
            arr,
            dims=[casealt_dim],
            coords={
                casealt_dim: model_dataset.coords[casealt_dim],
            },
            name=f"{s_tag}_data",
        )
    model_dataset = model_dataset.merge(da)

    if attach_indexes:
        altidx = datatree.root_dataset.coords[datatree.ALTIDX]
        altidx = altidx.drop_vars(list(altidx.coords))
        model_dataset[datatree.ALTIDX] = altidx
        model_dataset.dc.ALTIDX = datatree.ALTIDX
        caseptr = datatree.root_dataset[datatree.CASEPTR]
        caseptr = caseptr.drop_vars(list(caseptr.coords))
        model_dataset[datatree.CASEPTR] = caseptr
        model_dataset.dc.CASEPTR = datatree.CASEPTR
        model_dataset = model_dataset.assign_coords(
            {
                datatree.CASEID: DataArray(
                    datatree.caseids(),
                    dims=(datatree.CASEID),
                ),
                datatree.ALTID: DataArray(
                    datatree.altids(),
                    dims=(datatree.ALTID),
                ),
            }
        )
    model_dataset.dc.CASEID = datatree.CASEID
    model_dataset.dc.ALTID = datatree.ALTID
    return model_dataset, flow


def _prep_co(
    model_dataset,
    shared_data_co,
    vars_co,
    tag="co",
    preserve_vars=True,
    dtype=None,
    dim_name=None,
    cache_dir=None,
    flow=None,
    use_array_maker=True,
    use_eval=True,
):
    from ..dataset import DataArray, DataTree

    assert isinstance(shared_data_co, DataTree)
    if not isinstance(vars_co, dict):
        vars_co = {i: i for i in vars_co}
    if not vars_co:
        return model_dataset, None
    if use_eval:
        arr = shared_data_co.eval_many(
            vars_co, dtype=dtype, result_type="dataarray", with_coords=False
        ).values
    else:
        flowname = flownamer(tag, vars_co, shared_data_co._hash_features())
        if flow is None or flowname != flow.name:
            flow = shared_data_co.setup_flow(
                vars_co, cache_dir=cache_dir, name=flowname
            )
        arr = flow.load(
            shared_data_co,
            dtype=dtype,
            use_array_maker=use_array_maker,
        )
    caseid_dim = shared_data_co.CASEID
    if preserve_vars or len(vars_co) > 1:
        if dim_name is None:
            dim_name = f"var_{tag}"
        if arr.size:
            arr = arr.reshape(
                model_dataset.sizes.get(caseid_dim),
                -1,
            )
        else:
            arr = arr.reshape(
                model_dataset.sizes.get(caseid_dim),
                len(vars_co.keys()),
            )
        da = DataArray(
            arr,
            dims=[caseid_dim, dim_name],
            coords={
                caseid_dim: model_dataset.coords[caseid_dim],
                dim_name: list(vars_co.keys()),
            },
            name=tag,
        )
    else:
        arr = arr.reshape(
            model_dataset.sizes.get(caseid_dim),
        )
        da = DataArray(
            arr,
            dims=[caseid_dim],
            coords={
                caseid_dim: model_dataset.coords[caseid_dim],
            },
            name=tag,
        )
    return model_dataset.merge(da), flow


@nb.njit
def ce_to_dense(ce_data, ce_altidx, ce_caseptr, n_alts):
    if ce_caseptr.ndim == 2:
        ce_caseptr1 = ce_caseptr[:, -1]
    else:
        ce_caseptr1 = ce_caseptr[1:]
    shape = (ce_caseptr1.shape[0], n_alts, *ce_data.shape[1:])
    out = np.zeros(shape, dtype=ce_data.dtype)
    c = 0
    for row in range(ce_data.shape[0]):
        if row == ce_caseptr1[c]:
            c += 1
        a = ce_altidx[row]
        out[c, a, ...] = ce_data[row, ...]
    return out


@nb.njit
def ce_bool_to_dense(ce_altidx, ce_caseptr, n_alts):
    if ce_caseptr.ndim == 2:
        ce_caseptr1 = ce_caseptr[:, -1]
    else:
        ce_caseptr1 = ce_caseptr[1:]
    shape = (ce_caseptr1.shape[0], n_alts)
    out = np.zeros(shape, dtype=np.int8)
    c = 0
    for row in range(ce_altidx.shape[0]):
        if row == ce_caseptr1[c]:
            c += 1
        a = ce_altidx[row]
        out[c, a, ...] = 1
    return out

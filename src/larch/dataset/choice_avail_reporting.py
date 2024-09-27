from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype


def clean_summary(df, root_id=None):
    if root_id is not None:
        totals = df.loc[root_id, :]
        df.drop(index=root_id, inplace=True)
    else:
        totals = df.sum()

    relevant_totals = (
        "chosen",
        "chosen weighted",
        "chosen unweighted",
        "chosen but not available",
        "chosen but not available weighted",
        "chosen but not available unweighted",
        "chosen thus available",
        "not available so not chosen",
    )

    for tot in relevant_totals:
        if tot in totals:
            df.loc["< Total All Alternatives >", tot] = totals[tot]

    for i in df.columns:
        if i not in relevant_totals:
            if is_object_dtype(df[i].dtype) or is_string_dtype(df[i].dtype):
                df.loc["< Total All Alternatives >", i] = ""
    df.drop("_root_", errors="ignore", inplace=True)

    if "availability condition" in df:
        df["availability condition"] = df["availability condition"].fillna("")

    for i in (
        "chosen",
        "chosen but not available",
        "chosen thus available",
        "not available so not chosen",
    ):
        if i in df.columns and all(df[i] == df[i].astype(int)):
            df[i] = df[i].astype(int)

    for i in ("available",):
        if i in df.columns:
            df[i] = df[i].astype(pd.Int64Dtype())

    return df


def choice_avail_summary(
    dataset, graph=None, availability_co_vars=None, streaming=None
):
    """
    Generate a summary of choice and availability statistics.

    Parameters
    ----------
    dataset : Dataset
        The loaded dataset to summarize, which should have
        `ch` and `av` variables.
    graph : NestingTree, optional
        The nesting graph.
    availability_co_vars : dict, optional
        Also attach the definition of the availability conditions.

    Returns
    -------
    pandas.DataFrame
    """
    if graph is None:
        if "ch" in dataset:
            ch_ = np.asarray(dataset["ch"].copy())
        else:
            ch_ = None
        av_ = np.asarray(dataset.get("av"))
    else:
        from ..model.cascading import array_av_cascade, array_ch_cascade

        ch_ = array_ch_cascade(dataset.get("ch"), graph)
        av_ = array_av_cascade(dataset.get("av"), graph)
        av_[av_ > 1] = 1

    if ch_ is not None:
        ch = ch_.sum(0)
    else:
        ch = None

    if av_ is not None:
        av = av_.sum(0)
    else:
        av = None

    arr_wt = dataset.get("wt")
    if arr_wt is not None:
        if ch_ is not None:
            ch_w = pd.Series((ch_ * arr_wt.values.reshape(-1, 1)).sum(0))
        else:
            ch_w = None
        if av_ is not None:
            av_w = pd.Series((av_ * arr_wt.values.reshape(-1, 1)).sum(0))
        else:
            av_w = None
        show_wt = np.any(ch != ch_w)
    else:
        ch_w = ch
        av_w = av
        show_wt = False

    if av_ is not None:
        ch_[av_ > 0] = 0
    if ch_ is not None and ch_.sum() > 0:
        ch_but_not_av = ch_.sum(0)
        if arr_wt is not None:
            ch_but_not_av_w = pd.Series((ch_ * arr_wt.values).sum(0), index=ch_.columns)
        else:
            ch_but_not_av_w = ch_but_not_av
    else:
        ch_but_not_av = None
        ch_but_not_av_w = None

    from collections import OrderedDict

    od = OrderedDict()

    if graph is not None:
        idx = graph.standard_sort
        od["name"] = pd.Series(graph.standard_sort_names, index=idx)
    else:
        idx = dataset.dc.altids()
        if dataset.get("alt_names") is not None:
            od["name"] = pd.Series(dataset.get("alt_names"), index=idx)
        elif dataset.get("altnames") is not None:
            od["name"] = pd.Series(dataset.get("altnames"), index=idx)

    if show_wt:
        od["chosen weighted"] = pd.Series(ch_w, index=idx)
        od["chosen unweighted"] = pd.Series(ch, index=idx)
        od["available weighted"] = pd.Series(av_w, index=idx)
        od["available unweighted"] = pd.Series(av, index=idx)
    else:
        od["chosen"] = pd.Series(ch, index=idx)
        od["available"] = pd.Series(av, index=idx)
    if ch_but_not_av is not None:
        if show_wt:
            od["chosen but not available weighted"] = pd.Series(
                ch_but_not_av_w, index=idx
            )
            od["chosen but not available unweighted"] = pd.Series(
                ch_but_not_av, index=idx
            )
        else:
            od["chosen but not available"] = pd.Series(ch_but_not_av, index=idx)

    if availability_co_vars is not None:
        od["availability condition"] = pd.Series(
            availability_co_vars.values(),
            index=availability_co_vars.keys(),
            dtype=np.str_,
        )

    result = pd.DataFrame.from_dict(od)
    result = clean_summary(result, root_id=graph.root_id if graph is not None else None)
    # if graph is not None:
    #     totals = result.loc[graph.root_id, :]
    #     result.drop(index=graph.root_id, inplace=True)
    # else:
    #     totals = result.sum()
    #
    # for tot in (
    #     "chosen",
    #     "chosen weighted",
    #     "chosen unweighted",
    #     "chosen but not available",
    #     "chosen but not available weighted",
    #     "chosen but not available unweighted",
    #     "chosen thus available",
    #     "not available so not chosen",
    # ):
    #     if tot in totals:
    #         result.loc["< Total All Alternatives >", tot] = totals[tot]
    #
    # result.loc[
    #     "< Total All Alternatives >",
    #     pd.isnull(result.loc["< Total All Alternatives >", :]),
    # ] = ""
    # result.drop("_root_", errors="ignore", inplace=True)
    #
    # if "availability condition" in result:
    #     result["availability condition"] = result["availability condition"].fillna("")
    #
    # for i in (
    #     "chosen",
    #     "chosen but not available",
    #     "chosen thus available",
    #     "not available so not chosen",
    # ):
    #     if i in result.columns and all(result[i] == result[i].astype(int)):
    #         result[i] = result[i].astype(int)
    #
    # for i in ("available",):
    #     if i in result.columns:
    #         j = result.columns.get_loc(i)
    #         if all(result.iloc[:-1, j] == result.iloc[:-1, j].astype(int)):
    #             result.iloc[:-1, j] = result.iloc[:-1, j].astype(int)

    return result

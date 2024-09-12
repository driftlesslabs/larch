from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from ..util import dictx
from .numbamodel import NumbaModel as Model

logger = logging.getLogger(__name__)


def doctor(
    model: Model,
    repair_ch_av: Literal["?", "+", "-", None] = "?",
    repair_ch_zq=None,
    repair_asc=None,
    repair_noch_nzwt: Literal["?", "+", "-", None] = "?",
    repair_nan_wt=None,
    repair_nan_data_co: Literal["?", True, "!", None] = "?",
    check_low_variance_data_co: Literal["?", "!", None] = None,
    verbose=3,
    warning_stacklevel=2,
):
    problems = dictx()

    if not isinstance(model, Model):
        raise TypeError("the doctor requires a Model instance to diagnose")

    # logger.info("checking for chosen-but-zero-quantity")
    # model, diagnosis = chosen_but_zero_quantity(
    #     model, repair=repair_ch_zq, verbose=verbose
    # )
    # if diagnosis is not None:
    #     logger.warning(
    #         f"problem: chosen-but-zero-quantity ({len(diagnosis)} issues)",
    #         stacklevel=warning_stacklevel + 1,
    #     )
    #     problems["chosen_but_zero_quantity"] = diagnosis

    def apply_repair(repair, repair_func):
        nonlocal model, problems, verbose
        if repair is None:
            return
        logger.info(f"checking for {repair_func.__name__}")
        model, diagnosis = repair_func(model, repair=repair, verbose=verbose)
        if diagnosis is not None:
            logger.warning(
                f"problem: {repair_func.__name__} has ({len(diagnosis)} issues)",
                stacklevel=warning_stacklevel + 2,
            )
            problems[repair_func.__name__] = diagnosis

    apply_repair(repair_ch_av, chosen_but_not_available)
    apply_repair(repair_noch_nzwt, nothing_chosen_but_nonzero_weight)
    apply_repair(repair_nan_data_co, nan_data_co)
    apply_repair(check_low_variance_data_co, low_variance_data_co)

    # logger.info("checking for nan-weight")
    # model, diagnosis = nan_weight(model, repair=repair_nan_wt, verbose=verbose)
    # if diagnosis is not None:
    #     logger.warning(
    #         f"problem: nan-weight ({len(diagnosis)} issues)",
    #         stacklevel=warning_stacklevel + 1,
    #     )
    #     problems["nan_weight"] = diagnosis
    #
    # logger.info("checking for nan-data_co")
    # model, diagnosis = nan_data_co(model, repair=repair_nan_data_co, verbose=verbose)
    # if diagnosis is not None:
    #     logger.warning("problem: nan-data_co", stacklevel=warning_stacklevel + 1)
    #     problems["nan_data_co"] = diagnosis
    #
    # logger.info("checking for low-variance-data-co")
    # model, diagnosis = low_variance_data_co(model, repair=None, verbose=verbose)
    # if diagnosis is not None:
    #     logger.warning(
    #         f"problem: low-variance-data-co ({len(diagnosis)} issues)",
    #         stacklevel=warning_stacklevel + 1,
    #     )
    #     problems["low_variance_data_co"] = diagnosis

    return model, problems


def chosen_but_not_available(
    model: Model, repair: Literal["?", "+", "-", "!"] = "?", verbose: int = 3
) -> tuple[Model, pd.DataFrame | None]:
    """
    Check if some observations are chosen but not available.

    Parameters
    ----------
    model : larch.Model
        The model to check.
    repair : {None, '+', '-'}
        How to repair the data.
        Plus will make the conflicting alternatives available.
        Minus will make them not chosen (possibly leaving no chosen alternative).
        A question mark effects no repair, and simply emits a warning. An exclamation
        mark will raise an error if there are any conflicts.
    verbose : int, default 3
        The number of example rows to list for each problem.

    Returns
    -------
    model : larch.Model
        The model with revised dataset attached.
    diagnosis : pd.DataFrame
        The number of bad instances, by alternative, and some example rows.

    """
    dataset = model.dataset
    if dataset is None:
        raise ValueError("data not loaded")
    assert isinstance(dataset, xr.Dataset)

    not_avail = dataset["av"] == 0
    chosen = dataset["ch"] > 0
    chosen_but_not_available = not_avail & chosen
    chosen_but_not_available_sum = chosen_but_not_available.sum(dataset.dc.CASEID)

    diagnosis = None
    if chosen_but_not_available_sum.sum() > 0:
        i1, i2 = np.where(chosen_but_not_available)

        diagnosis = pd.DataFrame(
            chosen_but_not_available_sum[chosen_but_not_available_sum > 0],
            columns=[
                "n",
            ],
        )

        for colnum, colname in enumerate(
            chosen_but_not_available.coords[dataset.dc.ALTID]
        ):
            if chosen_but_not_available_sum[colnum] > 0:
                diagnosis.loc[str(colname), "example rows"] = ", ".join(
                    str(j) for j in i1[i2 == colnum][:verbose]
                )

        if repair == "+":
            model.dataset["av"].data[
                chosen_but_not_available.values[:, : model.dataset["av"].shape[1]]
            ] = 1
        elif repair == "-":
            model.dataset["ch"].data[
                chosen_but_not_available.values[:, : model.dataset["ch"].shape[1]]
            ] = 0
        elif repair == "!":
            raise ValueError("some observed choices are not available")

    return model, diagnosis


def chosen_but_zero_quantity(
    model, repair: Literal[None, "-"] = None, verbose: int = 3
):
    """
    Check if some observations are chosen but have zero quantity.

    Parameters
    ----------
    model : BaseModel
        The model to check
    repair : {None, '-', }
        How to repair the data.
        Minus will make them not chosen (possibly leaving no chosen alternative).
        None effects no repair, and simply emits a warning.
    verbose : int, default 3
        The number of example rows to list for each problem.

    Returns
    -------
    model : BaseModel
        The model with revised data
    diagnosis : pd.DataFrame
        The number of bad instances, by alternative, and some example rows.

    """
    if repair not in ("-", None):
        raise ValueError(f'invalid repair setting "{repair}"')

    if isinstance(model, Model):
        m = model
        model = m.dataframes
    else:
        m = None

    try:
        zero_q = model.get_zero_quantity_ca()
    except ValueError:
        diagnosis = None

    else:
        _zero_q = (zero_q > 0).values
        _chosen = (model.data_ch > 0).values
        _wid = min(_zero_q.shape[1], _chosen.shape[1])

        chosen_but_zero_quantity = pd.DataFrame(
            data=_zero_q[:, :_wid] & _chosen[:, :_wid],
            index=model.data_av.index,
            columns=model.data_av.columns[:_wid],
        )
        chosen_but_zero_quantity_sum = chosen_but_zero_quantity.sum(0)

        diagnosis = None
        if chosen_but_zero_quantity_sum.sum() > 0:
            i1, i2 = np.where(chosen_but_zero_quantity)

            diagnosis = pd.DataFrame(
                chosen_but_zero_quantity_sum[chosen_but_zero_quantity_sum > 0],
                columns=[
                    "n",
                ],
            )

            for colnum, colname in enumerate(chosen_but_zero_quantity.columns):
                if chosen_but_zero_quantity_sum[colname] > 0:
                    diagnosis.loc[colname, "example rows"] = ", ".join(
                        str(j) for j in i1[i2 == colnum][:verbose]
                    )

            if repair == "-":
                model.data_ch.values[chosen_but_zero_quantity] = 0

    if m is None:
        return model, diagnosis
    else:
        return m, diagnosis


def nothing_chosen_but_nonzero_weight(
    model, repair: Literal["?", "-", "*"] = "?", verbose=3
):
    """
    Check if some observations have no choice but have some weight.

    Parameters
    ----------
    model : BaseModel
        The model to check.
    repair : {None, '-', '*'}
        How to repair the data.
        Minus will make the weight zero when there is no choice. Star will
        also make the weight zero, plus autoscale all remaining weights.
        None effects no repair, and simply emits a warning.
    verbose : int, default 3
        The number of example rows to list for each problem.

    Returns
    -------
    model : BaseModel
        The revised dataframe
    diagnosis : pd.DataFrame
        The number of bad instances, by alternative, and some example rows.

    """
    diagnosis = None
    dataset = model.dataset
    if dataset is None:
        raise ValueError("data not loaded")
    assert isinstance(dataset, xr.Dataset)

    if "wt" in dataset and "ch" in dataset:
        nothing_chosen = dataset["ch"].sum(dataset.dc.ALTID) == 0
        nothing_chosen_some_weight = nothing_chosen & (dataset["wt"] > 0)
        if nothing_chosen_some_weight.sum() > 0:
            i1 = np.where(nothing_chosen_some_weight)[0]

            diagnosis = pd.DataFrame(
                [
                    nothing_chosen_some_weight.sum(),
                ],
                columns=[
                    "n",
                ],
                index=[
                    "nothing_chosen_some_weight",
                ],
            )

            diagnosis.loc["nothing_chosen_some_weight", "example rows"] = ", ".join(
                str(j) for j in i1[:verbose]
            )
            if repair == "+":
                raise ValueError(
                    "cannot resolve nothing_chosen_but_nonzero_weight by assuming some choice"
                )
            elif repair == "-":
                model.dataset["wt"].data[nothing_chosen] = 0
            elif repair == "*":
                model.dataset["wt"].data[nothing_chosen] = 0
                model.dataset.dc.autoscale_weights()
    return model, diagnosis


def nan_weight(dataset, repair=None, verbose=3):
    """
    Check if some observations are chosen but not available.

    Parameters
    ----------
    dataset : DataFrames or Model
            The data to check
    repair : None or bool
            Whether to repair the data.
            Any true value will make NaN values in the weight zero.
            None effects no repair, and simply emits a warning.
    verbose : int, default 3
            The number of example rows to list for each problem.

    Returns
    -------
    dfs : DataFrames
            The revised dataframe
    diagnosis : pd.DataFrame
            The number of bad instances, and some example rows.

    """
    if isinstance(dataset, Model):
        m = dataset
        dataset = m.dataframes
    else:
        m = None

    diagnosis = None
    if dataset.data_wt is not None:
        nan_wgt = np.isnan(dataset.data_wt.iloc[:, 0])

        if nan_wgt.sum():
            i = np.where(nan_wgt)[0]

            diagnosis = pd.DataFrame(
                data=[[nan_wgt.sum(), ""]],
                columns=["n", "example rows"],
                index=["nan_weight"],
            )

            diagnosis.loc["nan_weight", "example rows"] = ", ".join(
                str(j) for j in i[:verbose]
            )

        if repair:
            dataset.data_wt.fillna(0, inplace=True)

    if m is None:
        return dataset, diagnosis
    else:
        return m, diagnosis


def nan_data_co(
    model: Model, repair: Literal["?", True, "!"] = "?", verbose: int = 3
) -> tuple[Model, pd.DataFrame | None]:
    """
    Check if some data_co values are NaN.

    Parameters
    ----------
    model : larch.Model
        The model to check.
    repair : {"?", "!", True}
        Whether to repair the data. Any true value other than "?" or "!" will make
        NaN values in data_co zero. The question mark simply emits a warning
        if there are NaN values found, while the exclamation mark will raise an error.
    verbose : int, default 3
        The number of example columns to list for each problem.

    Returns
    -------
    model : larch.Model
        The model with revised dataset attached.
    diagnosis : pd.DataFrame
        The number of bad instances, and some example rows.
    """
    dataset = model.dataset
    if dataset is None:
        raise ValueError("data not loaded")
    assert isinstance(dataset, xr.Dataset)

    diagnosis = None
    if "co" in dataset:
        nan_dat = np.isnan(dataset["co"]).sum(dataset.dc.CASEID)
        if nan_dat.sum():
            diagnosis = (
                nan_dat[nan_dat > 0]
                .iloc[:verbose]
                .to_pandas()
                .rename("n_nan")
                .to_frame()
            )
            n = int(nan_dat.sum())
            if repair == "?":
                logger.warning(f"nan_data_co: {n} instances have NaN values")
            elif repair == "!":
                raise ValueError(f"nan_data_co: {n} instances have NaN values")
            if repair and repair != "?":
                dataset["co"] = dataset["co"].fillna(0)

    return model, diagnosis


def low_variance_data_co(
    model: Model, repair: Literal["?", "!"] = "?", verbose: int = 3
):
    """
    Check if any data_co columns have very low variance.

    Parameters
    ----------
    model : larch.Model
        The model to check.
    repair : {"?", "!"}
        No repairs are available for this check. The question mark simply emits
        a warning if there are issues found, while the exclamation mark will
        raise an error.
    verbose : int, default 3
        The number of example columns to list if there is a problem.

    Returns
    -------
    model : larch.Model
        The model with revised dataset attached.
    diagnosis : pd.DataFrame
        The number of bad instances, and some example rows.
    """
    dataset = model.dataset
    if dataset is None:
        raise ValueError("data not loaded")
    assert isinstance(dataset, xr.Dataset)

    diagnosis = None
    if "co" in dataset:
        variance = dataset["co"].var(dataset.dc.CASEID).to_pandas().rename("variance")
        if variance.min() < 1e-3:
            diagnosis = variance[variance < 1e-3].to_frame()
            if repair == "?":
                logger.warning(
                    f"low_variance_data_co: {len(diagnosis)} columns have low variance"
                )
            elif repair == "!":
                raise ValueError(
                    f"low_variance_data_co: {len(diagnosis)} columns have low variance"
                )

    return model, diagnosis

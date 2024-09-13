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
    repair_ch_av: Literal["?", "+", "-", "!"] | None = "?",
    repair_ch_zq: Literal["?", "-", "!"] | None = None,
    repair_asc=None,
    repair_noch_nzwt: Literal["?", "+", "-"] | None = "?",
    repair_nan_wt: Literal["?", True, "!"] | None = "?",
    repair_nan_data_co: Literal["?", True, "!"] | None = "?",
    check_low_variance_data_co: Literal["?", "!"] | None = None,
    verbose=3,
    warning_stacklevel=2,
):
    problems = dictx()

    if not isinstance(model, Model):
        raise TypeError("the doctor requires a Model instance to diagnose")

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
    apply_repair(repair_nan_wt, nan_weight)
    apply_repair(check_low_variance_data_co, low_variance_data_co)
    apply_repair(repair_ch_zq, chosen_but_zero_quantity)

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
    model: Model, repair: Literal["?", "-", "!"] | None = None, verbose: int = 3
):
    """
    Check if some observations are chosen but have zero quantity.

    Parameters
    ----------
    model : BaseModel
        The model to check
    repair : {None, '-', '?', '!'}
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
    if repair not in ("?", "-", "!", None):
        raise ValueError(f'invalid repair setting "{repair}"')

    quant = model.quantity()
    zero_quantity = np.asarray(quant[:, : model.graph.n_elementals()] == 0)

    dataset = model.dataset
    if dataset is None:
        raise ValueError("data not loaded")
    assert isinstance(dataset, xr.Dataset)

    ch_and_zero_quantity = (dataset["ch"] > 0) & zero_quantity
    ch_and_zero_quantity_sum = ch_and_zero_quantity.sum(dataset.dc.CASEID)
    diagnosis = None
    if ch_and_zero_quantity.sum() > 0:
        i1, i2 = np.where(ch_and_zero_quantity)
        diagnosis = (
            ch_and_zero_quantity_sum[ch_and_zero_quantity_sum > 0]
            .to_pandas()
            .rename("n")
            .to_frame()
        )

        for colnum, colname in enumerate(dataset.dc.altids()):
            if ch_and_zero_quantity_sum[colnum] > 0:
                diagnosis.loc[colname, "example rows"] = ", ".join(
                    str(j) for j in i1[i2 == colnum][:verbose]
                )

        msg = "chosen_but_zero_quantity: some observed choices have zero quantity.\n"
        try:
            from tabulate import tabulate
        except ImportError:
            msg += diagnosis.to_string()
        else:
            msg += tabulate(diagnosis, headers="keys", tablefmt="fancy_outline")
        if repair == "!":
            raise ValueError(msg)
        elif repair == "?":
            logger.warning(msg)
        elif repair == "-":
            logger.warning(
                msg.replace(
                    "some observed choices", "zeroing out observed choices that"
                )
            )
            model.dataset["ch"].values[ch_and_zero_quantity] = 0

    return model, diagnosis


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


def nan_weight(
    model: Model, repair: Literal["?", True, "!"] = "?", verbose: int = 3
) -> tuple[Model, pd.DataFrame | None]:
    """
    Check if some weight values are NaN.

    Parameters
    ----------
    model : larch.Model
        The model to check.
    repair : {"?", "!", True}
        Whether to repair the data. Any true value other than "?" or "!" will make
        NaN values in weight zero. The question mark simply emits a warning
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
    if "wt" in dataset:
        nan_wt = int(np.isnan(dataset["wt"]).sum())
        if nan_wt:
            diagnosis = (
                dataset["wt"][np.isnan(dataset["wt"])]
                .iloc[:verbose]
                .to_pandas()
                .to_frame()
            )
            if repair == "?":
                logger.warning(f"nan_weight: {nan_wt} instances have NaN values")
            elif repair == "!":
                raise ValueError(f"nan_weight: {nan_wt} instances have NaN values")
            if repair and repair != "?":
                dataset["wt"] = dataset["wt"].fillna(0)

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

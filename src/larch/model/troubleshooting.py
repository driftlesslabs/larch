from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from ..util import dictx
from .basemodel import BaseModel as Model

if TYPE_CHECKING:
    from .data_arrays import DataArrays

logger = logging.getLogger(__name__)


def doctor(
    model: Model,
    repair_ch_av="?",
    repair_ch_zq=None,
    repair_asc=None,
    repair_noch_nowt=None,
    repair_nan_wt=None,
    repair_nan_data_co=None,
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

    logger.info("checking for chosen-but-not-available")
    model, diagnosis = chosen_but_not_available(
        model, repair=repair_ch_av, verbose=verbose
    )
    if diagnosis is not None:
        logger.warning(
            f"problem: chosen-but-not-available ({len(diagnosis)} issues)",
            stacklevel=warning_stacklevel + 1,
        )
        problems["chosen_but_not_available"] = diagnosis
    #
    # logger.info("checking for nothing-chosen-but-nonzero-weight")
    # model, diagnosis = nothing_chosen_but_nonzero_weight(
    #     model, repair=repair_noch_nowt, verbose=verbose
    # )
    # if diagnosis is not None:
    #     logger.warning(
    #         f"problem: nothing-chosen-but-nonzero-weight ({len(diagnosis)} issues)",
    #         stacklevel=warning_stacklevel + 1,
    #     )
    #     problems["nothing_chosen_but_nonzero_weight"] = diagnosis
    #
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


def chosen_but_not_available(model, repair: Literal["?", "+", "-"] = "?", verbose=3):
    """
    Check if some observations are chosen but not available.

    Parameters
    ----------
    model : BaseModel
        The model to check.
    repair : {None, '+', '-'}
        How to repair the data.
        Plus will make the conflicting alternatives available.
        Minus will make them not chosen (possibly leaving no chosen alternative).
        A question mark effects no repair, and simply emits a warning.
    verbose : int, default 3
        The number of example rows to list for each problem.

    Returns
    -------
    model : BaseModel
        The model with revised data attached.
    diagnosis : pd.DataFrame
        The number of bad instances, by alternative, and some example rows.

    """
    data_arrays: DataArrays = model._data_arrays
    if data_arrays is None:
        raise ValueError("data not loaded")

    _not_avail = data_arrays.av == 0
    _chosen = data_arrays.ch > 0

    chosen_but_not_available = pd.DataFrame(
        data=_not_avail & _chosen,
        index=model.dataset.dc.caseids(),
        columns=model.graph.standard_sort,
    )
    chosen_but_not_available_sum = chosen_but_not_available.sum(0)

    diagnosis = None
    if chosen_but_not_available_sum.sum() > 0:
        i1, i2 = np.where(chosen_but_not_available)

        diagnosis = pd.DataFrame(
            chosen_but_not_available_sum[chosen_but_not_available_sum > 0],
            columns=[
                "n",
            ],
        )

        for colnum, colname in enumerate(chosen_but_not_available.columns):
            if chosen_but_not_available_sum[colname] > 0:
                diagnosis.loc[colname, "example rows"] = ", ".join(
                    str(j) for j in i1[i2 == colnum][:verbose]
                )

        if repair == "+":
            model.dataset["av"].data[
                chosen_but_not_available.values[:, : model.dataset["av"].shape[1]]
            ] = 1
            data_arrays.av[chosen_but_not_available.values] = 1
        elif repair == "-":
            model.dataset["ch"].data[
                chosen_but_not_available.values[:, : model.dataset["ch"].shape[1]]
            ] = 0
            data_arrays.ch[chosen_but_not_available.values] = 0
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


def nothing_chosen_but_nonzero_weight(dataset, repair=None, verbose=3):
    """
    Check if some observations have no choice but have some weight.

    Parameters
    ----------
    dataset : DataFrames or Model
        The data to check
    repair : {None, '-', '*'}
        How to repair the data.
        Minus will make the weight zero when there is no choice. Star will
        also make the weight zero, plus autoscale all remaining weights.
        None effects no repair, and simply emits a warning.
    verbose : int, default 3
        The number of example rows to list for each problem.

    Returns
    -------
    dfs : DataFrames
        The revised dataframe
    diagnosis : pd.DataFrame
        The number of bad instances, by alternative, and some example rows.

    """
    if isinstance(dataset, Model):
        m = dataset
        dataset = m.dataframes
    else:
        m = None

    diagnosis = None

    if dataset is None:
        raise ValueError("data not loaded")

    if dataset.data_wt is not None and dataset.data_ch is not None:
        nothing_chosen = dataset.array_ch().sum(1) == 0
        nothing_chosen_some_weight = nothing_chosen & (
            dataset.array_wt().reshape(-1) > 0
        )
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
                    "cannot resolve chosen_but_zero_quantity by assuming some choice"
                )
            elif repair == "-":
                dataset.array_wt()[nothing_chosen] = 0
            elif repair == "*":
                dataset.array_wt()[nothing_chosen] = 0
                dataset.autoscale_weights()
    if m is None:
        return dataset, diagnosis
    else:
        return m, diagnosis


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


def nan_data_co(dataset, repair=None, verbose=3):
    """
    Check if some data_co values are NaN.

    Parameters
    ----------
    dataset : DataFrames or Model
            The data to check
    repair : None or bool
            Whether to repair the data.
            Any true value will make NaN values in data_co zero.
            None effects no repair, and simply emits a warning.
    verbose : int, default 3
            The number of example columns to list for each problem.

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
    if dataset.data_co is not None:
        nan_dat = np.isnan(dataset.data_co).sum()
        if nan_dat.sum():
            diagnosis = pd.DataFrame(nan_dat[nan_dat > 0].iloc[:verbose])
        if repair:
            dataset.data_co.fillna(0, inplace=True)

    if m is None:
        return dataset, diagnosis
    else:
        return m, diagnosis


def low_variance_data_co(dataset, repair=None, verbose=3):
    """
    Check if any data_co columns have very low variance.

    Parameters
    ----------
    dataset : DataFrames or Model
            The data to check
    repair : None
            Not implemented.
    verbose : int, default 3
            The number of example columns to list if
            there is a problem.

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
    if dataset.data_co is not None:
        variance = dataset.data_co.var()
        if variance.min() < 1e-3:
            i = np.where(variance < 1e-3)[0]

            diagnosis = pd.DataFrame(
                data=[[len(i), ""]],
                columns=["n", "example cols"],
                index=["low_variance_co"],
            )

            diagnosis.loc["low_variance_co", "example cols"] = ", ".join(
                str(dataset.data_co.columns[j]) for j in i[:verbose]
            )

    if m is None:
        return dataset, diagnosis
    else:
        return m, diagnosis

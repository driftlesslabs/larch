from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from ..util import dictx
from .basemodel import BaseModel
from .numbamodel import NumbaModel as Model
from .possible_overspec import (
    PossibleOverspecificationError,
    compute_possible_overspecification,
)

logger = logging.getLogger(__name__)


def doctor(
    model: Model,
    *,
    repair_ch_av: Literal["?", "+", "-", "!"] | None = "?",
    repair_ch_zq: Literal["?", "-", "!"] | None = None,
    repair_av_zq: Literal["?", "-", "!"] | None = None,
    repair_noch_nzwt: Literal["?", "+", "-"] | None = "?",
    repair_nan_wt: Literal["?", True, "!"] | None = "?",
    repair_nan_data_co: Literal["?", True, "!"] | None = "?",
    check_low_variance_data_co: Literal["?", "!"] | None = None,
    check_overspec: Literal["?", "!", None] = None,
    repair_nan_utility: Literal["?", True, "!"] | None = "?",
    verbose: int = 3,
    warning_stacklevel: int = 2,
):
    """
    Diagnose data problems with a model.

    The doctor will check for common data problems that can cause numerical
    instability in a model.  The doctor will return a list of problems found,
    and optionally repair them.

    Parameters
    ----------
    model : larch.Model
        The model to diagnose.
    repair_ch_av : {'?', '+', '-', '!'}, default '?'
        How to repair the data if some observations are chosen but not available.
        The plus ('+') will make the conflicting alternatives available, overriding
        the availability status. The minus ('-') will make them not chosen (possibly
        leaving no chosen alternative). A question mark ('?') effects no repair, and
        simply emits a warning without interrupting program execution. An exclamation
        mark will raise an error if there are any conflicts.
    repair_ch_zq : {'?', '-', '!'}, default None
        How to repair the data if some observations are chosen but have zero quantity.
        The minus ('-') will make alternatives with zero quantity not chosen (possibly
        leaving no chosen alternative). A question mark ('?') effects no repair, and
        simply emits a warning. An exclamation mark ('!') will raise an error if there
        are any conflicts.
    repair_av_zq : {'?', '-', '!'}, default None
        How to repair the data if some observations are available but have zero quantity.
        The minus ('-') will make alternatives with zero quantity not available (possibly
        leaving no available alternatives). A question mark ('?') effects no repair, and
        simply emits a warning. An exclamation mark ('!') will raise an error if there are
        any conflicts.
    repair_noch_nzwt : {'?', '+', '-'}, default '?'
        How to repair the data if some observations have no choice but have some weight.
        Minus ('-') will make the weight zero when there is no choice. Plus ('+') will
        make the weight zero, plus autoscale all remaining weights so the total of the
        case weights equals the number of cases. A question mark ('?') effects no repair,
        and simply emits a warning.
    repair_nan_wt : {'?', '!', True}, default '?'
        How to repair the data if some weight values are NaN. Any true value other than
        "?" or "!" will make NaN values in weight zero. The question mark simply emits
        a warning if there are NaN values found, while the exclamation mark will raise
        an error.
    repair_nan_data_co : {'?', '!', True}, default '?'
        How to repair the data if some data_co values are NaN. Any true value other than
        "?" or "!" will make NaN values in data_co zero. The question mark simply emits
        a warning if there are NaN values found, while the exclamation mark will raise
        an error.
    check_low_variance_data_co : {'?', '!'}, default None
        Check if any data_co columns have very low variance. No repairs are available for
        this check. The question mark simply emits a warning if there are issues found,
        while the exclamation mark will raise an error.
    check_overspec : {'?', '!'}, default None
        Check model for possible over-specification. No repairs are available for this
        check. A question mark ('?') simply emits a warning if a possible over-
        specification is found. An exclamation mark ('!') will raise an error if
        possible over-specification is found.  This is considered a "deep" check, and
        will only be run if there are no known data problems found by the other checks.
    repair_nan_utility : {'?', '!', True}, default '?'
        How to repair the data if some utility values are NaN at current parameters.
        Any true value other than "?" or "!" will take alternatives with NaN values in
        utility, and make them unavailable. The question mark simply emits a warning if
        there are NaN values found, while the exclamation mark will raise an error. This
        is considered a "deep" check, and will only be run if there are no known data
        problems found by the other checks.
    verbose : int, default 3
        The number of example rows to list for each problem.
    warning_stacklevel : int, default 2
        The stacklevel for warnings.

    Returns
    -------
    model : larch.Model
        The model with revised dataset attached.
    problems : dict
        A dictionary of problems found, with the key being the name of the problem
        and the value being a DataFrame with the number of bad instances and some
        example rows.

    Raises
    ------
    TypeError
        If the model is not a Model instance.
    ValueError
        If any of the repair settings are invalid, or if the repair is set to '!' and
        there are any conflicts found.
    """
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
    apply_repair(repair_av_zq, available_but_zero_quantity)

    if not problems:
        # Following are deep checks, which require actually evaluating the model.
        # We run these after the data checks above, so we can skip this work if there
        # are already known data problems.
        apply_repair(repair_nan_utility, nan_utility)
        apply_repair(check_overspec, overspecification)

    return model, problems


def chosen_but_not_available(
    model: Model, repair: Literal["?", "+", "-", "!"] = "?", verbose: int = 3
) -> tuple[Model, pd.DataFrame | None]:
    """
    Check if some observations are chosen but not available.

    Alternatives that are unavailable have their utility values set to negative
    infinity.  If even one observation is chosen but not available, the model
    log-likelihood will nominally be negative infinity regardless of the values
    of any other parameters.  Note that some compute engines (e.g. JAX) may
    not actually return negative infinity log likelihoods due to clipping of
    extreme values.

    Parameters
    ----------
    model : larch.Model
        The model to check.
    repair : {'?', '+', '-', '!'}, default '?'
        How to repair the data. The plus ('+') will make the conflicting
        alternatives available, overriding the availability status. The minus
        ('-') will make them not chosen (possibly leaving no chosen alternative).
        A question mark ('?') effects no repair, and simply emits a warning
        without interrupting program execution. An exclamation mark will raise
        an error if there are any conflicts.
    verbose : int, default 3
        The number of example rows to list for each problem.

    Returns
    -------
    model : larch.Model
        The model with revised dataset attached.
    diagnosis : pd.DataFrame
        The number of bad instances, by alternative, and some example rows.

    Raises
    ------
    ValueError
        If the repair is set to '!' and there are any conflicts found.
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
        diagnosis.insert(0, "altid", None)

        diagnosis_rownum = 0
        for colnum, colname in enumerate(
            chosen_but_not_available.coords[dataset.dc.ALTID]
        ):
            if chosen_but_not_available_sum[colnum] > 0:
                diagnosis.loc[diagnosis_rownum, "example rows"] = ", ".join(
                    str(j) for j in i1[i2 == colnum][:verbose]
                )
                if isinstance(colname, xr.DataArray) and colname.ndim == 0:
                    colname = colname.item()
                diagnosis.loc[diagnosis_rownum, "altid"] = colname
                diagnosis_rownum += 1

        if repair == "+":
            model.dataset["av"].data[
                chosen_but_not_available.values[:, : model.dataset["av"].shape[1]]
            ] = 1
        elif repair == "-":
            model.dataset["ch"].data[
                chosen_but_not_available.values[:, : model.dataset["ch"].shape[1]]
            ] = 0
        elif repair == "!":
            raise ValueError(
                "some observed choices are not available (try `repair_ch_av`)"
            )

    return model, diagnosis


def chosen_but_zero_quantity(
    model: Model, repair: Literal["?", "-", "!"] | None = None, verbose: int = 3
):
    """
    Check if some observations are chosen but have zero quantity.

    Alternatives that have zero quantity have utility values that end up as
    negative infinity, regardless of whether the alternative would otherwise be
    available.  Due to the mathematical structure of how quantities are used in
    Larch, this situation is generally a result of a data problem, and not
    due to the current values of model parameters.

    If even one observation is chosen but has zero quantity, the model
    log-likelihood will nominally be negative infinity regardless of the values
    of any other parameters.  Note that some compute engines (e.g. JAX) may
    not actually return negative infinity log likelihoods due to clipping of
    extreme values.

    Parameters
    ----------
    model : BaseModel
        The model to check.
    repair : {'?', '-', '!'}
        How to repair the data. The minus ('-') will make alternatives with zero
        quantity not chosen (possibly leaving no chosen alternative). A question
        mark ('?') effects no repair, and simply emits a warning. An exclamation
        mark ('!') will raise an error if there are any conflicts.
    verbose : int, default 3
        The number of example rows to list for each problem.

    Returns
    -------
    model : BaseModel
        The model with revised data
    diagnosis : pd.DataFrame
        The number of bad instances, by alternative, and some example rows.

    Raises
    ------
    ValueError
        If the repair is set to '!' and there are any conflicts found.
    """
    if repair not in ("?", "-", "!", None):
        raise ValueError(f'invalid repair setting "{repair}"')

    if not model.quantity_ca:
        # no quantities, so no problem
        return model, None

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
            msg += "\nTry `repair_ch_zq` to resolve."
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


def available_but_zero_quantity(
    model: Model, repair: Literal["?", "-", "!"] | None = None, verbose: int = 3
):
    """
    Check if some observations are available but have zero quantity.

    Alternatives that have zero quantity have utility values that end up as
    negative infinity, regardless of whether the alternative would otherwise be
    available.  Due to the mathematical structure of how quantities are used in
    Larch, this situation is generally a result of a data problem, and not
    due to the current values of model parameters.

    If even one observation is available but has zero quantity, the first
    derivative of the model log-likelihood may be incalculable, and the model
    parameter estimation process may fail.

    Parameters
    ----------
    model : BaseModel
        The model to check.
    repair : {'?', '-', '!'}
        How to repair the data. The minus ('-') will make alternatives with zero
        quantity not available (possibly leaving no available alternatives). A
        question mark ('?') effects no repair, and simply emits a warning. An
        exclamation mark ('!') will raise an error if there are any conflicts.
    verbose : int, default 3
        The number of example rows to list for each problem.

    Returns
    -------
    model : BaseModel
        The model with revised data
    diagnosis : pd.DataFrame
        The number of bad instances, by alternative, and some example rows.

    Raises
    ------
    ValueError
        If the repair is set to '!' and there are any conflicts found.
    """
    if repair not in ("?", "-", "!", None):
        raise ValueError(f'invalid repair setting "{repair}"')

    if not model.quantity_ca:
        # no quantities, so no problem
        return model, None

    quant = model.quantity()
    zero_quantity = np.asarray(quant[:, : model.graph.n_elementals()] == 0)

    dataset = model.dataset
    if dataset is None:
        raise ValueError("data not loaded")
    assert isinstance(dataset, xr.Dataset)

    av_and_zero_quantity = (dataset["av"] > 0) & zero_quantity
    av_and_zero_quantity_sum = av_and_zero_quantity.sum(dataset.dc.CASEID)
    diagnosis = None
    if av_and_zero_quantity.sum() > 0:
        i1, i2 = np.where(av_and_zero_quantity)
        diagnosis = (
            av_and_zero_quantity_sum[av_and_zero_quantity_sum > 0]
            .to_pandas()
            .rename("n")
            .to_frame()
        )

        for colnum, colname in enumerate(dataset.dc.altids()):
            if av_and_zero_quantity_sum[colnum] > 0:
                diagnosis.loc[colname, "example rows"] = ", ".join(
                    str(j) for j in i1[i2 == colnum][:verbose]
                )

        msg = (
            "available_but_zero_quantity: some available choices have zero quantity.\n"
        )
        try:
            from tabulate import tabulate
        except ImportError:
            msg += diagnosis.to_string()
        else:
            msg += tabulate(diagnosis, headers="keys", tablefmt="fancy_outline")
        if repair == "!":
            msg += "\nTry `repair_av_zq` to resolve."
            raise ValueError(msg)
        elif repair == "?":
            logger.warning(msg)
        elif repair == "-":
            logger.warning(
                msg.replace(
                    "some available choices", "zeroing out available choices that"
                )
            )
            model.dataset["av"].values[av_and_zero_quantity] = 0

    return model, diagnosis


def nothing_chosen_but_nonzero_weight(
    model, repair: Literal["?", "-", "*", "!"] = "?", verbose=3
):
    """
    Check if some observations have no choice but have some weight.

    Parameters
    ----------
    model : BaseModel
        The model to check.
    repair : {'?', '-', '*', '!'}
        How to repair the data. Minus ('-') will make the weight zero when there
        is no choice. Star ('*') will also make the weight zero, plus autoscale
        all remaining weights so the total of the case weights equals the number
        of cases. A question mark ('?') effects no repair, and simply emits a
        warning.
    verbose : int, default 3
        The number of example rows to list for each problem.

    Returns
    -------
    model : BaseModel
        The revised dataframe
    diagnosis : pd.DataFrame
        The number of bad instances, by alternative, and some example rows.

    Raises
    ------
    ValueError
        If the repair is set to '!' and there are any conflicts found.
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
            msg = "nothing_chosen_but_nonzero_weight: some cases have no choice but non-zero weight.\n"
            try:
                from tabulate import tabulate
            except ImportError:
                msg += diagnosis.to_string()
            else:
                msg += tabulate(diagnosis, headers="keys", tablefmt="fancy_outline")
            if repair == "!":
                msg += "\nTry `repair_noch_nzwt` to resolve."
                raise ValueError(msg)
            elif repair == "?":
                logger.warning(msg)
            elif repair == "+":
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

    Raises
    ------
    ValueError
        If the repair is set to '!' and there are any NaN values found.
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
                raise ValueError(
                    f"nan_data_co: {n} instances have NaN values, try `repair_nan_data_co`"
                )
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

    Raises
    ------
    ValueError
        If the repair is set to '!' and there are any NaN values found.
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
                raise ValueError(
                    f"nan_weight: {nan_wt} instances have NaN values, try `repair_nan_wt`"
                )
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

    Raises
    ------
    ValueError
        If the repair is set to '!' and there are any low variance columns found.
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


def nan_utility(model: Model, repair: Literal["?", True, "!"] = "?", verbose: int = 3):
    """
    Check if any utility values are NaN at current parameters.

    Parameters
    ----------
    model : larch.Model
        The model to check.
    repair : {'?', '!', True}
        Whether to repair the data. Any true value other than "?" or "!" will
        take alternatives with NaN values in utility, and make them unavailable.
        The question mark simply emits a warning if there are NaN values found,
        while the exclamation mark will raise an error.
    verbose : int, default 3
        The number of example columns to list for each problem.

    Returns
    -------
    model : larch.Model
        The model with revised dataset attached.
    diagnosis : pd.DataFrame
        The number of bad instances, and some example rows.

    Raises
    ------
    ValueError
        If the repair is set to '!' and there are any NaN values found.
    """
    dataset = model.dataset
    if dataset is None:
        raise ValueError("data not loaded")
    assert isinstance(dataset, xr.Dataset)

    n_alts = dataset["av"].shape[1]

    u = model.utility()[:, :n_alts]
    nan_u = np.isnan(u) & (dataset["av"] > 0)

    diagnosis = None
    nan_util = int(nan_u.sum())
    if nan_util:
        diagnosis = nan_u.sum(dataset.dc.CASEID).to_pandas().rename("n").to_frame()
        if repair == "?":
            logger.warning(
                f"nan_utility: {nan_util} available alternatives have NaN "
                f"utility values"
            )
        elif repair == "!":
            raise ValueError(
                f"nan_utility: {nan_util} available alternatives have NaN "
                f"utility values,\ntry using the Model.doctor `repair_nan_utility` "
                f"argument, set it to True to make them unavailable"
            )
        elif repair:
            model.dataset["av"].data[nan_u] = 0

    return model, diagnosis


def overspecification(
    model: BaseModel, repair: Literal["?", "!"] = "?", verbose: int = 3
):
    """
    Check model for possible over-specification.

    Parameters
    ----------
    model : larch.Model
        The model to check.
    repair : {'?', '!'}
        No automatic repairs are available for this check. A question mark ('?') simply
        emits a warning if a possible over-specification is found. An exclamation mark
        ('!') will raise an error if possible over-specification is found.
    verbose : int, default 3
        This is ignored for the overspecification check; all possible problems are
        listed.

    Returns
    -------
    model : larch.Model
        The model with revised dataset attached.
    diagnosis : pd.DataFrame
        A dataframe of possible over-specification problems in the model.  The index of
        this dataframe is a multi-index with the first level being the problem number, the
        second level being the eigenvalue, and the third level being the parameter name[s]
        of the non-zero elements of each problematic eigenvector. The columns are the
        non-zero eigenvector values.
    """
    pvals = model.pvals
    locks = np.asarray(model.pholdfast.astype(bool))
    if model.compute_engine == "jax":
        _se, hess, _inv_hess = model.jax_param_cov(pvals)
    else:
        hess = -model.d2_loglike(pvals)
    hess = np.asarray(hess).copy()
    hess[locks, :] = 0
    hess[:, locks] = 0

    diagnosis = None
    overspec = compute_possible_overspecification(hess, model.pholdfast)
    if overspec:
        diagnosis = []
        possible_overspecification = []
        msg = "Model is possibly over-specified (hessian is nearly singular)."
        msg += "\nLook for problems in these parameters or groups of parameters:"
        for eigval, ox, eigenvec in overspec:
            if eigval == "LinAlgError":
                possible_overspecification.append((eigval, [ox], [""]))
            else:
                paramset = list(np.asarray(model.pnames)[ox])
                possible_overspecification.append((eigval, paramset, eigenvec[ox]))
                diagnosis.append(
                    (
                        eigval,
                        pd.Series(eigenvec[ox], index=paramset, name="eigenvector"),
                    )
                )
                msg += f"\n- Eigenvalue: {eigval}"
                max_len_param = max(len(p) for p in paramset)
                for p, z in zip(paramset, eigenvec[ox]):
                    msg += f"\n    {p:{max_len_param}s}: {z}"
        model._possible_overspecification = possible_overspecification
        if repair == "!":
            raise PossibleOverspecificationError(msg)
        elif repair == "?":
            logger.warning(msg)
        diagnosis = pd.concat(
            [d[1] for d in diagnosis],
            keys=[(n, d[0]) for (n, d) in enumerate(diagnosis)],
            names=["problem", "eigenvalue", "parameter"],
        )
        if isinstance(diagnosis, pd.Series):
            diagnosis = diagnosis.to_frame()
    return model, diagnosis

from __future__ import annotations

import logging
import pathlib
import warnings
from collections import namedtuple
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import xarray as xr
from numba import boolean, guvectorize, njit
from numba import float32 as f32
from numba import float64 as f64
from numba import int8 as i8
from numba import int32 as i32
from numba import int64 as i64

from ..dataset import DataArray, Dataset, DataTree
from ..exceptions import MissingDataError
from ..model.possible_overspec import (
    PossibleOverspecification,
    compute_possible_overspecification,
)
from ..util import dictx
from ..util.simple_attribute import SimpleAttribute
from .basemodel import BaseModel as _BaseModel
from .numba_stream import ModelStreamer

if TYPE_CHECKING:
    import altair.vegalite.v5.api

if not list(Path(__file__).parent.glob("__pycache__/numbamodel.*.nbc")):
    warnings.warn(  ## Good news, everyone! ##  )
        "\n\n"
        "#### larch v6 is experimental, and not feature-complete ####\n"
        "the first time you import on a new system, this package will\n"
        "compile optimized binaries for your specific machine,  which\n"
        "may take a little while, please be patient ...\n",
        stacklevel=1,
    )

logger = logging.getLogger(__package__)


@njit(cache=True)
def minmax(x):
    maximum = x[0]
    minimum = x[0]
    for i in x[1:]:
        if i > maximum:
            maximum = i
        elif i < minimum:
            minimum = i
    return (minimum, maximum)


@njit(cache=True)
def outside_range(x, bottom, top):
    for i in x[:]:
        if i == -np.inf:
            continue
        if i > top:
            return True
        elif i < bottom:
            return True
    return False


@njit(error_model="numpy", fastmath=True, cache=True)
def utility_from_data_co(
    model_utility_co_alt,  # int input shape=[n_co_features]
    model_utility_co_param_scale,  # float input shape=[n_co_features]
    model_utility_co_param,  # int input shape=[n_co_features]
    model_utility_co_data,  # int input shape=[n_co_features]
    parameter_arr,  # float input shape=[n_params]
    holdfast_arr,  # float input shape=[n_params]
    array_av,  # int8 input shape=[n_alts]
    data_co,  # float input shape=[n_co_vars]
    utility_elem,  # float output shape=[n_alts]
    dutility_elem,  # float output shape=[n_alts, n_params]
):
    for i in range(model_utility_co_alt.shape[0]):
        altindex = model_utility_co_alt[i]
        param_value = parameter_arr[model_utility_co_param[i]]
        param_holdfast = holdfast_arr[model_utility_co_param[i]]
        if array_av[altindex]:
            if model_utility_co_data[i] == -1:
                utility_elem[altindex] += param_value * model_utility_co_param_scale[i]
                if not param_holdfast:
                    dutility_elem[altindex, model_utility_co_param[i]] += (
                        model_utility_co_param_scale[i]
                    )
            else:
                _temp = (
                    data_co[model_utility_co_data[i]] * model_utility_co_param_scale[i]
                )
                utility_elem[altindex] += _temp * param_value
                if not param_holdfast:
                    dutility_elem[altindex, model_utility_co_param[i]] += _temp


@njit(error_model="numpy", fastmath=True, cache=True)
def quantity_from_data_ca(
    model_q_ca_param_scale,  # float input shape=[n_q_ca_features]
    model_q_ca_param,  # int input shape=[n_q_ca_features]
    model_q_ca_data,  # int input shape=[n_q_ca_features]
    model_q_scale_param,  # int input scalar
    parameter_arr,  # float input shape=[n_params]
    holdfast_arr,  # float input shape=[n_params]
    array_av,  # int8 input shape=[n_alts]
    array_ca,  # float input shape=[n_alts, n_ca_vars]
    utility_elem,  # float output shape=[n_alts]
    dutility_elem,  # float output shape=[n_alts, n_params]
):
    n_alts = array_ca.shape[0]

    if model_q_scale_param[0] >= 0:
        scale_param_value = parameter_arr[model_q_scale_param[0]]
        scale_param_holdfast = holdfast_arr[model_q_scale_param[0]]
    else:
        scale_param_value = 1.0
        scale_param_holdfast = 1

    for j in range(n_alts):
        # if self._array_ce_reversemap is not None:
        #     if c >= self._array_ce_reversemap.shape[0] or j >= self._array_ce_reversemap.shape[1]:
        #         row = -1
        #     else:
        #         row = self._array_ce_reversemap[c, j]
        _row = -1

        if array_av[j]:  # and row != -1:
            if model_q_ca_param.shape[0]:
                for i in range(model_q_ca_param.shape[0]):
                    # if row >= 0:
                    #     _temp = self._array_ce[row, self.model_quantity_ca_data[i]]
                    # else:
                    _temp = (
                        array_ca[j, model_q_ca_data[i]]
                        * model_q_ca_param_scale[i]
                        * np.exp(parameter_arr[model_q_ca_param[i]])
                    )
                    utility_elem[j] += _temp
                    if not holdfast_arr[model_q_ca_param[i]]:
                        dutility_elem[j, model_q_ca_param[i]] += (
                            _temp * scale_param_value
                        )

                for i in range(model_q_ca_param.shape[0]):
                    if not holdfast_arr[model_q_ca_param[i]]:
                        dutility_elem[j, model_q_ca_param[i]] /= utility_elem[j]

                _tempsize = np.log(utility_elem[j])
                utility_elem[j] = _tempsize * scale_param_value
                if (model_q_scale_param[0] >= 0) and not scale_param_holdfast:
                    dutility_elem[j, model_q_scale_param[0]] += _tempsize

        else:
            utility_elem[j] = -np.inf


@njit(error_model="numpy", fastmath=True, cache=True)
def utility_from_data_ca(
    model_utility_ca_param_scale,  # int input shape=[n_u_ca_features]
    model_utility_ca_param,  # int input shape=[n_u_ca_features]
    model_utility_ca_data,  # int input shape=[n_u_ca_features]
    parameter_arr,  # float input shape=[n_params]
    holdfast_arr,  # float input shape=[n_params]
    array_av,  # int8 input shape=[n_alts]
    array_ca,  # float input shape=[n_alts, n_ca_vars]
    array_ce_data,  # float input shape=[n_casealts, n_ca_vars]
    array_ce_indices,  # int input shape=[n_casealts]
    array_ce_ptr,  # int input shape=[2]
    utility_elem,  # float output shape=[n_alts]
    dutility_elem,  # float output shape=[n_alts, n_params]
):
    n_alts = array_ca.shape[0]

    if array_ce_data.shape[0] > 0:
        j = 0
        for row in range(array_ce_ptr[0], array_ce_ptr[1]):
            while array_ce_indices[row] > j:
                # skipped alts are unavail
                utility_elem[j] = -np.inf
                j += 1
            # j = array_ce_indices[row] # happens naturally
            for i in range(model_utility_ca_param.shape[0]):
                _temp = array_ce_data[row, model_utility_ca_data[i]]
                _temp *= model_utility_ca_param_scale[i]
                utility_elem[j] += _temp * parameter_arr[model_utility_ca_param[i]]
                if not holdfast_arr[model_utility_ca_param[i]]:
                    dutility_elem[j, model_utility_ca_param[i]] += _temp
            j += 1
        while n_alts > j:
            # skipped alts are unavail
            utility_elem[j] = -np.inf
            j += 1
    else:
        for j in range(n_alts):
            row = -1

            if array_av[j]:
                for i in range(model_utility_ca_param.shape[0]):
                    if row >= 0:
                        _temp = 0.0  # _temp = self._array_ce[row, self.model_utility_ca_data[i]]
                    else:
                        _temp = array_ca[j, model_utility_ca_data[i]]
                    _temp *= model_utility_ca_param_scale[i]
                    utility_elem[j] += _temp * parameter_arr[model_utility_ca_param[i]]
                    if not holdfast_arr[model_utility_ca_param[i]]:
                        dutility_elem[j, model_utility_ca_param[i]] += _temp
            else:
                utility_elem[j] = -np.inf


def _type_signature(sig, precision=32):
    result = ()
    for s in sig:
        if s == "f":
            result += (f32[:],) if precision == 32 else (f64[:],)
        elif s == "F":
            result += (f32[:, :],) if precision == 32 else (f64[:, :],)
        elif s == "r":
            result += (f32,) if precision == 32 else (f64,)
        elif s == "i":
            result += (i32[:],)
        elif s == "j":
            result += (i64[:],)
        elif s == "I":
            result += (i32[:, :],)
        elif s == "b":
            result += (i8[:],)
        elif s == "S":
            result += (i8,)
        elif s == "B":
            result += (boolean,)
    return result


def _type_signatures(sig):
    return [
        _type_signature(sig, precision=32),
        _type_signature(sig, precision=64),
    ]


@njit(error_model="numpy", fastmath=True, cache=True)
def _numba_utility_to_loglike(
    n_alts,
    edgeslots,  # int input shape=[edges, 4]
    mu_slots,  # int input shape=[nests]
    start_slots,  # int input shape=[nests]
    len_slots,  # int input shape=[nests]
    holdfast_arr,  # int8 input shape=[n_params]
    parameter_arr,  # float input shape=[n_params]
    array_ch,  # float input shape=[nodes]
    array_av,  # int8 input shape=[nodes]
    array_wt,  # float input shape=[]
    return_flags,  #
    dutility,  #
    utility,  # float output shape=[nodes]
    logprob,  # float output shape=[nodes]
    probability,  # float output shape=[nodes]
    bhhh,  # float output shape=[n_params, n_params]
    d_loglike,  # float output shape=[n_params]
    loglike,  # float output shape=[]
):
    assert edgeslots.shape[1] == 4
    upslots = edgeslots[:, 0]  # int input shape=[edges]
    dnslots = edgeslots[:, 1]  # int input shape=[edges]
    visit1 = edgeslots[:, 2]  # int input shape=[edges]
    _allocslot = edgeslots[:, 3]  # int input shape=[edges]

    assert return_flags.size == 4
    only_utility = return_flags[0]  # [19] int8 input
    return_probability = return_flags[1]  # [20] bool input
    return_grad = return_flags[2]  # [21] bool input
    return_bhhh = return_flags[3]  # [22] bool input

    # util_nx = np.zeros_like(utility)
    # mu_extra = np.zeros_like(util_nx)
    loglike[0] = 0.0

    if True:  # outside_range(utility[:n_alts], -0.0, 0.0):
        for up in range(n_alts, utility.size):
            up_nest = up - n_alts
            n_children_for_parent = len_slots[up_nest]
            shifter = -np.inf
            shifter_position = -1
            if mu_slots[up_nest] < 0:
                mu_up = 1.0
            else:
                mu_up = parameter_arr[mu_slots[up_nest]]
            if mu_up:
                for n in range(n_children_for_parent):
                    edge = start_slots[up_nest] + n
                    dn = dnslots[edge]
                    if utility[dn] > -np.inf:
                        z = utility[dn] / mu_up
                        if z > shifter:
                            shifter = z
                            shifter_position = dn
                        # TODO alphas
                        # if alpha[edge] > 0:
                        #     z = (logalpha[edge] + utility[child]) / mu[parent]
                        #     if z > shifter:
                        #         shifter = z
                        #         shifter_position = child
                for n in range(n_children_for_parent):
                    edge = start_slots[up_nest] + n
                    dn = dnslots[edge]
                    if utility[dn] > -np.inf:
                        if shifter_position == dn:
                            utility[up] += 1
                        else:
                            utility[up] += np.exp((utility[dn] / mu_up) - shifter)
                        # if alpha[edge] > 0:
                        #     if shifter_position == child:
                        #         utility[parent] += 1
                        #     else:
                        #         z = ((logalpha[edge] + utility[child]) / mu[parent]) - shifter
                        #         utility[parent] += exp(z)
                utility[up] = (np.log(utility[up]) + shifter) * mu_up
            else:  # mu_up is zero
                for n in range(n_children_for_parent):
                    edge = start_slots[up_nest] + n
                    dn = dnslots[edge]
                    if utility[dn] > utility[up]:
                        utility[up] = utility[dn]
    else:
        for s in range(upslots.size):
            dn = dnslots[s]
            up = upslots[s]
            up_nest = up - n_alts
            dn_nest = dn - n_alts
            if mu_slots[up_nest] < 0:
                mu_up = 1.0
            else:
                mu_up = parameter_arr[mu_slots[up_nest]]
            if visit1[s] > 0 and dn >= n_alts:
                log_dn = np.log(utility[dn])
                # mu_extra[dn] += log_dn + util_nx[dn]/utility[dn]
                if mu_slots[dn_nest] < 0:
                    mu_dn = 1.0
                else:
                    mu_dn = parameter_arr[mu_slots[dn_nest]]
                utility[dn] = log_dn * mu_dn
            util_dn = utility[dn]
            exp_util_dn_mu_up = np.exp(util_dn / mu_up)
            utility[up] += exp_util_dn_mu_up
            # util_nx[up] -= util_dn * exp_util_dn_mu_up / mu_up

        # mu_extra[mu_extra.size-1] += np.log(utility[utility.size-1])
        #                              + util_nx[-1]/utility[utility.size-1]
        utility[utility.size - 1] = np.log(utility[utility.size - 1])

    if only_utility == 2:
        return

    for s in range(upslots.size):
        dn = dnslots[s]
        up = upslots[s]
        if mu_slots[up - n_alts] < 0:
            mu_up = 1.0
        else:
            mu_up = parameter_arr[mu_slots[up - n_alts]]
        if np.isinf(utility[up]) and utility[up] < 0:
            logprob[dn] = -np.inf
        else:
            logprob[dn] = (utility[dn] - utility[up]) / mu_up
        if array_ch[dn]:
            loglike[0] += logprob[dn] * array_ch[dn] * array_wt[0]

    if return_probability or return_grad or return_bhhh:
        # logprob becomes conditional_probability
        conditional_probability = logprob
        for i in range(logprob.size):
            if array_av[i]:
                conditional_probability[i] = np.exp(logprob[i])

        # probability
        probability[-1] = 1.0
        for s in range(upslots.size - 1, -1, -1):
            dn = dnslots[s]
            if array_av[dn]:
                up = upslots[s]
                probability[dn] = probability[up] * conditional_probability[dn]
            else:
                probability[dn] = 0.0

        if return_grad or return_bhhh:
            d_loglike[:] = 0.0

            # d utility
            for s in range(upslots.size):
                dn = dnslots[s]
                up = upslots[s]
                if array_av[dn]:
                    cond_prob = conditional_probability[dn]
                    if dn >= n_alts:
                        dn_mu_slot = mu_slots[dn - n_alts]
                        if dn_mu_slot >= 0:
                            dutility[dn, dn_mu_slot] += utility[dn]
                            dutility[dn, dn_mu_slot] /= parameter_arr[dn_mu_slot]
                    up_mu_slot = mu_slots[up - n_alts]
                    if up_mu_slot >= 0:
                        # FIXME: alpha slots to appear here if cross-nesting is activated
                        dutility[up, up_mu_slot] -= cond_prob * (utility[dn])
                    dutility[up, :] += np.where(
                        cond_prob, cond_prob * dutility[dn, :], 0
                    )

            # d probability
            # scratch = np.zeros_like(parameter_arr)
            # d_probability = np.zeros_like(dutility)
            # for s in range(upslots.size-1, -1, -1):
            #     dn = dnslots[s]
            #     if array_ch[dn]:
            #         up = upslots[s]
            #         scratch[:] = dutility[dn] - dutility[up]
            #         up_mu_slot = mu_slots[up-n_alts]
            #         if up_mu_slot < 0:
            #             mu_up = 1.0
            #         else:
            #             mu_up = parameter_arr[up_mu_slot]
            #         if mu_up:
            #             if up_mu_slot >= 0:
            #                 scratch[up_mu_slot] += (utility[up] - utility[dn]) / mu_up
            #                 # FIXME: alpha slots to appear here if cross-nesting is activated
            #             multiplier = probability[up] / mu_up
            #         else:
            #             multiplier = 0
            #
            #         scratch[:] *= multiplier
            #         scratch[:] += d_probability[up, :]
            #         d_probability[dn, :] += scratch[:] *
            #           conditional_probability[dn] # FIXME: for CNL, use edge not dn

            # d probability alternate path slightly lower memory usage and some faster
            d_probability = np.zeros_like(dutility)
            for s in range(upslots.size - 1, -1, -1):
                dn = dnslots[s]
                if array_ch[dn]:
                    up = upslots[s]
                    up_mu_slot = mu_slots[up - n_alts]
                    if up_mu_slot < 0:
                        mu_up = 1.0
                    else:
                        mu_up = parameter_arr[up_mu_slot]
                    for p in range(parameter_arr.size):
                        if mu_up:
                            scratch_ = dutility[dn, p] - dutility[up, p]
                            if p == up_mu_slot:
                                scratch_ += (utility[up] - utility[dn]) / mu_up
                                # FIXME: alpha slots to appear here if cross-nesting is activated
                            scratch_ *= probability[up] / mu_up
                        else:
                            scratch_ = 0
                        scratch_ += d_probability[up, p]
                        if conditional_probability[dn]:
                            d_probability[dn, p] += (
                                scratch_ * conditional_probability[dn]
                            )  # FIXME: for CNL, use edge not dn

            if return_bhhh:
                bhhh[:] = 0.0

            # d loglike
            for a in range(n_alts):
                this_ch = array_ch[a]
                if this_ch == 0:
                    continue
                total_probability_a = probability[a]
                # if total_probability_a > 0:
                #     tempvalue = d_probability[a, :] / total_probability_a
                #     if return_bhhh:
                #         bhhh += np.outer(tempvalue,tempvalue) * this_ch * array_wt[0]
                #     d_loglike += tempvalue * array_wt[0]
                #
                if total_probability_a > 0:
                    if total_probability_a < 1e-250:
                        total_probability_a = 1e-250
                    tempvalue = d_probability[a, :] * (this_ch / total_probability_a)
                    dLL_temp = tempvalue / this_ch
                    d_loglike += tempvalue * array_wt[0]
                    if return_bhhh:
                        bhhh += np.outer(dLL_temp, dLL_temp) * this_ch * array_wt[0]


_master_shape_signature = (
    "(qca),(qca),(qca),(), "
    "(uca),(uca),(uca), "
    "(uco),(uco),(uco),(uco), "
    "(edges,four), "
    "(nests),(nests),(nests), "
    "(params),(params), "
    "(nodes),(nodes),(),(vco),(alts,vca), "
    "(ces,vce),(ces),(two),  "
    "(four)->"
    "(nodes),(nodes),(nodes),(params,params),(params),()"
)


def _numba_master(
    model_q_ca_param_scale,  # [0] float input shape=[n_q_ca_features]
    model_q_ca_param,  # [1] int input shape=[n_q_ca_features]
    model_q_ca_data,  # [2] int input shape=[n_q_ca_features]
    model_q_scale_param,  # [3] int input scalar
    model_utility_ca_param_scale,  # [4] float input shape=[n_u_ca_features]
    model_utility_ca_param,  # [5] int input shape=[n_u_ca_features]
    model_utility_ca_data,  # [6] int input shape=[n_u_ca_features]
    model_utility_co_alt,  # [ 7] int input shape=[n_co_features]
    model_utility_co_param_scale,  # [ 8] float input shape=[n_co_features]
    model_utility_co_param,  # [ 9] int input shape=[n_co_features]
    model_utility_co_data,  # [10] int input shape=[n_co_features]
    edgeslots,  # [11] int input shape=[edges, 4]
    mu_slots,  # [12] int input shape=[nests]
    start_slots,  # [13] int input shape=[nests]
    len_slots,  # [14] int input shape=[nests]
    holdfast_arr,  # [15] int8 input shape=[n_params]
    parameter_arr,  # [16] float input shape=[n_params]
    array_ch,  # [17] float input shape=[nodes]
    array_av,  # [18] int8 input shape=[nodes]
    array_wt,  # [19] float input shape=[]
    array_co,  # [20] float input shape=[n_co_vars]
    array_ca,  # [21] float input shape=[n_alts, n_ca_vars]
    array_ce_data,  # [22] float input shape=[n_casealts, n_ca_vars]
    array_ce_indices,  # [23] int input shape=[n_casealts]
    array_ce_ptr,  # [24] int input shape=[2]
    return_flags,
    # only_utility,        # [19] int8 input
    # return_probability,  # [20] bool input
    # return_grad,         # [21] bool input
    # return_bhhh,         # [22] bool input
    utility,  # [23] float output shape=[nodes]
    logprob,  # [24] float output shape=[nodes]
    probability,  # [25] float output shape=[nodes]
    bhhh,  # [26] float output shape=[n_params, n_params]
    d_loglike,  # [27] float output shape=[n_params]
    loglike,  # [28] float output shape=[]
):
    n_alts = array_ca.shape[0]

    # assert edgeslots.shape[1] == 4
    # upslots   = edgeslots[:,0]  # int input shape=[edges]
    # dnslots   = edgeslots[:,1]  # int input shape=[edges]
    # visit1    = edgeslots[:,2]  # int input shape=[edges]
    # allocslot = edgeslots[:,3]  # int input shape=[edges]

    assert return_flags.size == 4
    only_utility = return_flags[0]  # int8 input
    # return_probability = return_flags[1]    # bool input
    # return_grad = return_flags[2]           # bool input
    # return_bhhh = return_flags[3]           # bool input

    utility[:] = 0.0
    dutility = np.zeros((utility.size, parameter_arr.size), dtype=utility.dtype)

    quantity_from_data_ca(
        model_q_ca_param_scale,  # float input shape=[n_q_ca_features]
        model_q_ca_param,  # int input shape=[n_q_ca_features]
        model_q_ca_data,  # int input shape=[n_q_ca_features]
        model_q_scale_param,  # int input scalar
        parameter_arr,  # float input shape=[n_params]
        holdfast_arr,  # float input shape=[n_params]
        array_av,  # int8 input shape=[n_nodes]
        array_ca,  # float input shape=[n_alts, n_ca_vars]
        utility[:n_alts],  # float output shape=[n_alts]
        dutility[:n_alts],
    )

    if only_utility == 3:
        if model_q_scale_param[0] >= 0:
            scale_param_value = parameter_arr[model_q_scale_param[0]]
        else:
            scale_param_value = 1.0
        utility[:n_alts] = np.exp(utility[:n_alts] / scale_param_value)
        return

    utility_from_data_ca(
        model_utility_ca_param_scale,  # int input shape=[n_u_ca_features]
        model_utility_ca_param,  # int input shape=[n_u_ca_features]
        model_utility_ca_data,  # int input shape=[n_u_ca_features]
        parameter_arr,  # float input shape=[n_params]
        holdfast_arr,  # float input shape=[n_params]
        array_av,  # int8 input shape=[n_nodes]
        array_ca,  # float input shape=[n_alts, n_ca_vars]
        array_ce_data,  # float input shape=[n_casealts, n_ca_vars]
        array_ce_indices,  # int input shape=[n_casealts]
        array_ce_ptr,  # int input shape=[2]
        utility[:n_alts],  # float output shape=[n_alts]
        dutility[:n_alts],
    )

    utility_from_data_co(
        model_utility_co_alt,  # int input shape=[n_co_features]
        model_utility_co_param_scale,  # float input shape=[n_co_features]
        model_utility_co_param,  # int input shape=[n_co_features]
        model_utility_co_data,  # int input shape=[n_co_features]
        parameter_arr,  # float input shape=[n_params]
        holdfast_arr,  # float input shape=[n_params]
        array_av,  # int8 input shape=[n_nodes]
        array_co,  # float input shape=[n_co_vars]
        utility[:n_alts],  # float output shape=[n_alts]
        dutility[:n_alts],
    )

    if only_utility == 1:
        return

    _numba_utility_to_loglike(
        n_alts,
        edgeslots,  # int input shape=[edges, 4]
        mu_slots,  # int input shape=[nests]
        start_slots,  # int input shape=[nests]
        len_slots,  # int input shape=[nests]
        holdfast_arr,  # int8 input shape=[n_params]
        parameter_arr,  # float input shape=[n_params]
        array_ch,  # float input shape=[nodes]
        array_av,  # int8 input shape=[nodes]
        array_wt,  # float input shape=[]
        return_flags,
        dutility,
        utility,  # float output shape=[nodes]
        logprob,  # float output shape=[nodes]
        probability,  # float output shape=[nodes]
        bhhh,  # float output shape=[n_params, n_params]
        d_loglike,  # float output shape=[n_params]
        loglike,  # float output shape=[]
    )


_numba_master_vectorized = guvectorize(
    _type_signatures("fiii fii ifii I iii bf fbffF Fij b fffFff"),
    _master_shape_signature,
    nopython=True,
    fastmath=True,
    target="parallel",
    cache=True,
)(
    _numba_master,
)


@njit(cache=True)
def softplus(i, sharpness=10):
    cut = 10 / sharpness
    if i > cut:
        return i
    # elif i < -cut:
    #     return 0.0
    else:
        return np.log1p(np.exp(i * sharpness)) / sharpness


@njit(cache=True)
def d_softplus(i, sharpness=10):
    cut = 1000 / sharpness
    if i > cut:
        return 1.0
    # elif i < -cut:
    #     return 0.0
    else:
        return 1 / (1 + np.exp(-i * sharpness))


@guvectorize(
    [
        _type_signature("ffffffff", precision=32),
        _type_signature("ffffffff", precision=64),
    ],
    ("(params),(params),(params),(),()->(),(params),(params)"),
    cache=True,
    nopython=True,
)
def bounds_penalty(
    param_array,  # [] float input shape=[n_params]
    lower_bounds,  # [] float input shape=[n_params]
    upper_bounds,  # [] float input shape=[n_params]
    constraint_intensity,  # [] float input shape=[]
    constraint_sharpness,  # [] float input shape=[]
    penalty,  # [] float output shape=[]
    d_penalty,  # [] float output shape=[n_params]
    d_penalty_binding,  # [] float output shape=[n_params]
):
    # penalty = 0.0
    # d_penalty = np.zeros_like(param_array)
    # d_penalty_binding = np.zeros_like(param_array)
    penalty[0] = 0.0
    d_penalty[:] = 0.0
    d_penalty_binding[:] = 0.0
    for i in range(param_array.size):
        diff_threshold = np.minimum(upper_bounds[i] - lower_bounds[i], 0.0001)
        low_diff = lower_bounds[i] - param_array[i]
        low_penalty = -softplus(low_diff, constraint_sharpness[0])
        high_diff = param_array[i] - upper_bounds[i]
        high_penalty = -softplus(high_diff, constraint_sharpness[0])
        penalty[0] += (low_penalty + high_penalty) * constraint_intensity[0]
        if low_penalty:
            d_penalty[i] += (
                d_softplus(lower_bounds[i] - param_array[i], constraint_sharpness[0])
                * constraint_intensity[0]
            )
        if high_penalty:
            d_penalty[i] -= (
                d_softplus(param_array[i] - upper_bounds[i], constraint_sharpness[0])
                * constraint_intensity[0]
            )
        if np.absolute(high_diff) < diff_threshold:
            d_penalty_binding[i] -= 0.5 * constraint_intensity[0]
        elif np.absolute(low_diff) < diff_threshold:
            d_penalty_binding[i] += 0.5 * constraint_intensity[0]
    # return penalty, d_penalty, d_penalty_binding


def _numba_penalty(
    param_array,  # [] float input shape=[n_params]
    lower_bounds,  # [] float input shape=[n_params]
    upper_bounds,  # [] float input shape=[n_params]
    constraint_intensity,  # [] float input shape=[]
    constraint_sharpness,  # [] float input shape=[]
    bhhh,  # [] float output shape=[n_params, n_params]
    d_loglike,  # [] float output shape=[n_params]
    loglike,  # [] float output shape=[]
):
    penalty = 0.0
    d_penalty = np.zeros_like(d_loglike)
    for i in range(param_array.size):
        low_penalty = -softplus(
            lower_bounds[i] - param_array[i], constraint_sharpness[0]
        )
        high_penalty = -softplus(
            param_array[i] - upper_bounds[i], constraint_sharpness[0]
        )
        penalty += (low_penalty + high_penalty) * constraint_intensity[0]
        if low_penalty:
            d_penalty[i] += (
                d_softplus(lower_bounds[i] - param_array[i], constraint_sharpness[0])
                * constraint_intensity[0]
            )
        if high_penalty:
            d_penalty[i] -= (
                d_softplus(param_array[i] - upper_bounds[i], constraint_sharpness[0])
                * constraint_intensity[0]
            )
    loglike[0] += penalty
    d_loglike[:] += d_penalty
    bhhh[:] += np.outer(d_penalty, d_penalty)


_numba_penalty_vectorized = guvectorize(
    _type_signatures("fffffFff"),
    ("(params),(params),(params),(),()->(params,params),(params),()"),
    nopython=True,
    fastmath=True,
    target="parallel",
    cache=True,
)(
    _numba_penalty,
)


def model_co_slots(data_provider: Dataset, model: _BaseModel, dtype=np.float64):
    len_co = sum(len(_) for _ in model.utility_co.values())
    model_utility_co_alt = np.zeros([len_co], dtype=np.int32)
    model_utility_co_param_scale = np.ones([len_co], dtype=dtype)
    model_utility_co_param = np.zeros([len_co], dtype=np.int32)
    model_utility_co_data = np.zeros([len_co], dtype=np.int32)

    j = 0

    param_loc = {}
    for _n, _pname in enumerate(model.pnames):
        param_loc[_pname] = _n
    data_loc = {}
    if isinstance(data_provider, Dataset):
        if "var_co" in data_provider.indexes:
            for _n, _dname in enumerate(data_provider.indexes["var_co"]):
                data_loc[_dname] = _n
        alternative_codes = data_provider.indexes[data_provider.dc.ALTID]
    else:
        raise TypeError(
            f"data_provider must be DataFrames or Dataset not {type(data_provider)}"
        )

    for alt, func in model.utility_co.items():
        altindex = alternative_codes.get_loc(alt)
        for i in func:
            model_utility_co_alt[j] = altindex
            model_utility_co_param[j] = param_loc[str(i.param)]
            model_utility_co_param_scale[j] = i.scale
            if i.data == "1":
                model_utility_co_data[j] = -1
            else:
                model_utility_co_data[j] = data_loc[
                    str(i.data)
                ]  # self._data_co.columns.get_loc(str(i.data))
            j += 1

    return (
        model_utility_co_alt,
        model_utility_co_param_scale,
        model_utility_co_param,
        model_utility_co_data,
    )


def model_u_ca_slots(data_provider: Dataset, model: _BaseModel, dtype=np.float64):
    if isinstance(data_provider, Dataset):
        looker = lambda tag: data_provider.indexes["var_ca"].get_loc(str(tag))
    else:
        raise TypeError(
            f"data_provider must be DataFrames or Dataset not {type(data_provider)}"
        )
    len_model_utility_ca = len(model.utility_ca)
    model_utility_ca_param_scale = np.ones([len_model_utility_ca], dtype=dtype)
    model_utility_ca_param = np.zeros([len_model_utility_ca], dtype=np.int32)
    model_utility_ca_data = np.zeros([len_model_utility_ca], dtype=np.int32)
    for n, i in enumerate(model.utility_ca):
        model_utility_ca_param[n] = model.get_param_loc(i.param)
        model_utility_ca_data[n] = looker(i.data)
        model_utility_ca_param_scale[n] = i.scale
    return (
        model_utility_ca_param_scale,
        model_utility_ca_param,
        model_utility_ca_data,
    )


def model_q_ca_slots(data_provider: Dataset, model: _BaseModel, dtype=np.float64):
    if isinstance(data_provider, Dataset):
        looker = lambda tag: data_provider.indexes["var_ca"].get_loc(str(tag))
    else:
        raise TypeError(
            f"data_provider must be DataFrames or Dataset not {type(data_provider)}"
        )
    len_model_q_ca = len(model.quantity_ca)
    model_q_ca_param_scale = np.ones([len_model_q_ca], dtype=dtype)
    model_q_ca_param = np.zeros([len_model_q_ca], dtype=np.int32)
    model_q_ca_data = np.zeros([len_model_q_ca], dtype=np.int32)
    if model.quantity_scale:
        model_q_scale_param = np.zeros([1], dtype=np.int32) + model.get_param_loc(
            model.quantity_scale
        )
    else:
        model_q_scale_param = np.zeros([1], dtype=np.int32) - 1
    for n, i in enumerate(model.quantity_ca):
        model_q_ca_param[n] = model.get_param_loc(i.param)
        model_q_ca_data[n] = looker(i.data)
        model_q_ca_param_scale[n] = i.scale
    return (
        model_q_ca_param_scale,
        model_q_ca_param,
        model_q_ca_data,
        model_q_scale_param,
    )


class _case_slice:
    def __get__(self, obj, objtype=None):
        self.parent = obj
        return self

    def __getitem__(self, idx):
        return type(self.parent)(
            **{k: getattr(self.parent, k)[idx] for k in self.parent._fields}
        )


WorkArrays = namedtuple(
    "WorkArrays",
    ["utility", "logprob", "probability", "bhhh", "d_loglike", "loglike"],
)
WorkArrays.cs = _case_slice()


FixedArrays = namedtuple(
    "FixedArrays",
    [
        "qca_scale",
        "qca_param_slot",
        "qca_data_slot",
        "qscale_param_slot",
        "uca_scale",
        "uca_param_slot",
        "uca_data_slot",
        "uco_alt_slot",
        "uco_scale",
        "uco_param_slot",
        "uco_data_slot",
        "edge_slots",
        "mu_slot",
        "start_edges",
        "len_edges",
    ],
)


class NumbaModel(_BaseModel):
    _null_slice = (None, None, None)
    streaming = ModelStreamer()
    constraint_intensity = SimpleAttribute(float)
    constraint_sharpness = SimpleAttribute(float)

    def __init__(self, *args, float_dtype=np.float64, datatree=None, **kwargs):
        for a in args:
            if datatree is None and isinstance(a, (DataTree, Dataset)):
                datatree = a
        super().__init__(datatree=datatree, **kwargs)
        self._dataset = None
        self._fixed_arrays = None
        self._data_arrays = None
        self.work_arrays = None
        self.float_dtype = float_dtype
        self.constraint_intensity = 0.0
        self.constraint_sharpness = 0.0
        self._constraint_funcs = None
        self.datatree = datatree
        self._should_preload_data = True
        cache_dir = kwargs.get("cache_dir", None)
        if cache_dir is not None:
            cache_dir = pathlib.Path(cache_dir)
            cache_dir.mkdir(exist_ok=True)
            self.datatree.cache_dir = cache_dir

        # when the alternative codes are set, and include zero, we need to
        # make change the root node id to be not zero.
        if isinstance(self.datatree, (DataTree, Dataset)):
            if 0 in self.datatree.dc.altids():
                # check that -1 is not in the altids
                if -1 in self.datatree.dc.altids():
                    raise ValueError("Cannot have both -1 and 0 as alternative codes")
                self.initialize_graph(root_id=-1)

    def save(self, filename, format="yaml", overwrite=False):
        from .saving import save_model

        return save_model(self, filename, format=format, overwrite=overwrite)

    def dumps(self):
        return repr(self.save(None, format="raw"))

    def should_preload_data(self, should=True):
        should = bool(should)
        if should and not self._should_preload_data:
            # not currently, but want to, mangle to prevent inconsistency
            self.mangle()
        self._should_preload_data = should

    @classmethod
    def from_dict(cls, content):
        self = cls()

        def loadthis(attr, wrapper=None, injector=None):
            i = content.get(attr, None)
            if i is not None:
                try:
                    if wrapper is not None:
                        i = wrapper(i)
                except AttributeError:
                    pass
                else:
                    if injector is None:
                        setattr(self, attr, i)
                    else:
                        injector(i)

        loadthis("float_dtype", lambda i: getattr(np, i))
        loadthis("compute_engine")
        loadthis("index_name")
        loadthis("parameters", xr.Dataset.from_dict, self.update_parameters)
        loadthis("availability_any")
        loadthis("availability_ca_var")
        loadthis("availability_co_vars")
        loadthis("choice_any")
        loadthis("choice_ca_var")
        loadthis("choice_co_code")
        loadthis("choice_co_vars")
        loadthis("common_draws")
        loadthis("constraint_intensity")
        loadthis("constraint_sharpness")
        loadthis("constraints")
        from .tree import NestingTree

        loadthis("graph", NestingTree.from_dict)
        loadthis("groupid")
        loadthis("logsum_parameter")
        loadthis("mixtures", self.mixtures.from_list)
        loadthis("n_draws")
        loadthis("prerolled_draws")
        loadthis("quantity_ca")
        loadthis("quantity_scale")
        loadthis("title")
        loadthis("utility_ca")
        loadthis("utility_co")
        loadthis("weight_co_var")
        loadthis("weight_normalization")
        return self

    work_arrays = SimpleAttribute()

    def mangle(self, data=True, structure=True) -> None:
        super().mangle(data, structure)
        if data:
            self._dataset = None
            self._data_arrays = None
            self.work_arrays = None
            self._array_ch_cascade = None
            self._array_av_cascade = None
        if structure:
            self._constraint_funcs = None
            self._fixed_arrays = None
            # print("self.streaming", self.streaming)
            self.streaming = None

    def is_mnl(self):
        """
        Check if this model is a MNL model.

        Returns
        -------
        bool
        """
        if self._graph is None:
            return True
        if len(self._graph) - len(self._graph.elementals) == 1:
            return True
        return False

    dataflows = SimpleAttribute(dict)

    def reflow_data_arrays(self) -> None:
        """Reload the internal data_arrays so they are consistent with the datatree."""
        if self.graph is None:
            self._data_arrays = None
            return

        if not self.use_streaming:
            logger.debug("Model.reflow_data_arrays with full datatree")
            datatree = self.datatree
        else:
            logger.debug(
                "Model.reflow_data_arrays with partial datatree, one case only"
            )
            datatree = self.datatree.replace_datasets(
                {
                    self.datatree.root_node_name: self.datatree.root_dataset.isel(
                        {self.datatree.CASEID: slice(0, 1)}
                    )
                }
            )
        if datatree is not None:
            from .data_arrays import prepare_data

            logger.debug(f"Model.datatree.cache_dir = {datatree.cache_dir}")
            self.dataset, self.dataflows = prepare_data(
                datasource=datatree,
                request=self,
                float_dtype=self.float_dtype,
                cache_dir=datatree.cache_dir,
                flows=self.dataflows,
                make_unused_flows=self.use_streaming,
            )
            if self.use_streaming:
                # when streaming the dataset created above is a vestigial
                # one-case dataset, really we just want the flows, so we
                # get rid of the dataset now
                self._dataset = None
                self._data_arrays = None
            else:
                if self.autoscale_weights:
                    self.dataset.dc.autoscale_weights()
                self._rebuild_data_arrays()
                if self.work_arrays is not None:
                    self._rebuild_work_arrays()

    def _rebuild_data_arrays(self):
        if self._dataset is None:
            self.reflow_data_arrays()  # create the dataset from the datatree
        else:
            self._data_arrays = self._dataset.dc.to_arrays(
                self.graph,
                float_dtype=self.float_dtype,
            )

    def _rebuild_work_arrays(
        self, n_cases=None, n_nodes=None, n_params=None, on_missing_data="silent"
    ):
        log = logging.getLogger("Larch")
        if n_cases is None:
            try:
                n_cases = self.n_cases
            except MissingDataError:
                if on_missing_data != "silent":
                    log.error("MissingDataError, cannot rebuild work arrays")
                self.work_arrays = None
                if on_missing_data == "raise":
                    raise
                return
        if n_nodes is None:
            n_nodes = len(self.graph)
        if n_params is None:
            n_params = self.n_params
        _need_to_rebuild_work_arrays = True
        if self.work_arrays is not None:
            if (
                (self.work_arrays.utility.shape[0] == n_cases)
                and (self.work_arrays.utility.shape[1] == n_nodes)
                and (self.work_arrays.d_loglike.shape[1] == n_params)
                and (self.work_arrays.utility.dtype == self.float_dtype)
            ):
                _need_to_rebuild_work_arrays = False
        if _need_to_rebuild_work_arrays:
            log.debug("rebuilding work arrays")
            self.work_arrays = WorkArrays(
                utility=np.zeros([n_cases, n_nodes], dtype=self.float_dtype),
                logprob=np.zeros([n_cases, n_nodes], dtype=self.float_dtype),
                probability=np.zeros([n_cases, n_nodes], dtype=self.float_dtype),
                bhhh=np.zeros([n_cases, n_params, n_params], dtype=self.float_dtype),
                d_loglike=np.zeros([n_cases, n_params], dtype=self.float_dtype),
                loglike=np.zeros([n_cases], dtype=self.float_dtype),
            )

    def _rebuild_fixed_arrays(self):
        data_provider = self.data_as_loaded
        if data_provider is not None:
            (
                model_utility_ca_param_scale,
                model_utility_ca_param,
                model_utility_ca_data,
            ) = model_u_ca_slots(data_provider, self, dtype=self.float_dtype)
            (
                model_utility_co_alt,
                model_utility_co_param_scale,
                model_utility_co_param,
                model_utility_co_data,
            ) = model_co_slots(data_provider, self, dtype=self.float_dtype)
            (
                model_q_ca_param_scale,
                model_q_ca_param,
                model_q_ca_data,
                model_q_scale_param,
            ) = model_q_ca_slots(data_provider, self, dtype=self.float_dtype)
            node_slot_arrays = self.graph.node_slot_arrays(self)
            n_alts = self.graph.n_elementals()
            self._fixed_arrays = FixedArrays(
                model_q_ca_param_scale,
                model_q_ca_param,
                model_q_ca_data,
                # (np.asarray([model_q_scale_param])
                # if self.use_streaming
                # else model_q_scale_param),
                model_q_scale_param,
                model_utility_ca_param_scale,
                model_utility_ca_param,
                model_utility_ca_data,
                model_utility_co_alt,
                model_utility_co_param_scale,
                model_utility_co_param,
                model_utility_co_data,
                np.stack(self.graph.edge_slot_arrays()).T,
                node_slot_arrays[0][n_alts:],
                node_slot_arrays[1][n_alts:],
                node_slot_arrays[2][n_alts:],
            )
        else:
            self._fixed_arrays = None

    def doctor(self, **kwargs):
        """
        Run diagnostics, checking for common problems and inconsistencies.

        See :func:`larch.model.troubleshooting.doctor` for more information.
        """
        result = super().doctor(**kwargs)
        self._rebuild_data_arrays()
        return result

    def unmangle(self, force=False, structure_only=False):
        if not self._mangled and not force:
            return
        marker = f"_currently_unmangling_{__file__}"
        if getattr(self, marker, False):
            return
        try:
            setattr(self, marker, True)
            super().unmangle(force=force)
            if not structure_only:
                if self._dataset is None or force:
                    self.reflow_data_arrays()
                if self._fixed_arrays is None or force:
                    self._rebuild_fixed_arrays()
                    self._rebuild_work_arrays()
                if self._constraint_funcs is None:
                    self._constraint_funcs = [
                        c.as_soft_penalty() for c in self.constraints
                    ]
        finally:
            delattr(self, marker)

    def _scan_logsums_ensure_names(self):
        nameset = set()
        try:
            g = self._graph
        except ValueError:
            pass
        else:
            if g is not None:
                for nodecode in g.topological_sorted_no_elementals:
                    if nodecode != g._root_id:
                        param_name = str(g.nodes[nodecode]["parameter"])
                        nameset.add(str(param_name))
        if self.quantity_ca is not None and len(self.quantity_ca) > 0:
            if self.quantity_scale is not None:
                nameset.add(str(self.quantity_scale))
        if self.logsum_parameter is not None:
            nameset.add(str(self.logsum_parameter))
        self._ensure_names(
            nameset, value=1, nullvalue=1, initvalue=1, minimum=0.01, maximum=1
        )

    def __prepare_for_compute(
        self,
        x=None,
        allow_missing_ch=False,
        allow_missing_av=False,
        caseslice=None,
    ):
        if caseslice is None:
            caseslice = slice(caseslice)
        if self.datatree is None and self.dataset is None:
            raise MissingDataError("dataset and datatree are both not set")
        self.unmangle()
        if x is not None:
            self.pvals = x
        if self.dataset is not None:
            if "ch" not in self.dataset and not allow_missing_ch:
                raise MissingDataError("model.dataset does not include `ch`")
        if self.work_arrays is None:
            self._rebuild_work_arrays(on_missing_data="raise")
        return (
            *self._fixed_arrays,
            self.pholdfast,
            self.pvals.astype(self.float_dtype),  # float input shape=[n_params]
            *self._data_arrays.cs[caseslice],  # TODO fix when not using named tuple
        )

    def constraint_violation(
        self,
        on_violation="raise",
        intensity_check=False,
    ):
        """
        Check if constraints are currently violated.

        Parameters
        ----------
        on_violation : {'raise', 'return'}
            If set to 'raise', an exception is raised if any model constraint,
            including any bound constraint, is violated.  Otherwise, this method
            returns a message describing the first constraint violation found, or
            an empty string if no violation are found.
        intensity_check : bool, default False
            If True, when the model's `constraint_intensity` attribute is set
            to zero, this function always returns OK (empty string).

        Returns
        -------
        str
            If no exception is raised, this method returns a message describing
            the first constraint violation found, or an empty string if no
            violation are found.
        """
        OK = ""
        if intensity_check and self.constraint_intensity == 0:
            return OK
        over_max = self.pvals > self.pmaximum
        if np.any(over_max):
            failure = np.where(over_max)[0][0]
            failure_message = (
                f"{self.pnames[failure]} over maximum "
                f"({self.pvals[failure]} > {self.pmaximum[failure]})"
            )
            if on_violation != "raise":
                return failure_message
            raise ValueError(failure_message)
        under_min = self.pvals < self.pminimum
        if np.any(under_min):
            failure = np.where(under_min)[0][0]
            failure_message = (
                f"{self.pnames[failure]} under minimum "
                f"({self.pvals[failure]} < {self.pminimum[failure]})"
            )
            if on_violation != "raise":
                return failure_message
            raise ValueError(failure_message)
        for c in self.constraints:
            if c.fun(self.pvals) < 0:
                failure_message = str(c)
                if on_violation != "raise":
                    return failure_message
                raise ValueError(failure_message)
        return OK

    def fit_bhhh(self, *args, **kwargs):
        from .numba_optimization import fit_bhhh

        return fit_bhhh(self, *args, **kwargs)

    def constraint_penalty(self, x=None):
        if x is not None:
            self.pvals = x
        penalty, dpenalty, dpenalty_binding = bounds_penalty(
            self.pvals.astype(self.float_dtype),
            self.pminimum.astype(self.float_dtype),
            self.pmaximum.astype(self.float_dtype),
            self.float_dtype(self.constraint_intensity),
            self.float_dtype(self.constraint_sharpness),
        )
        for cf, dcf, dcf_bind in self._constraint_funcs:
            penalty += cf(
                self.pvals,
                self.constraint_intensity,
                self.constraint_sharpness,
            )
            dpenalty += dcf(
                self.pvals,
                self.constraint_intensity,
                self.constraint_sharpness,
            )
            dpenalty_binding += dcf_bind(
                self.pvals,
                self.constraint_intensity,
            )
        return penalty, dpenalty, dpenalty_binding

    def constraint_converge_tolerance(self, x=None):
        args = self.__prepare_for_compute(
            x,
            allow_missing_ch=False,
        )
        args_flags = args + (
            np.asarray(
                [
                    0,  # only_utility
                    False,  # return_probability
                    True,  # return_gradient
                    True,  # return_bhhh
                ],
                dtype=np.int8,
            ),
        )
        with np.errstate(
            divide="ignore",
            over="ignore",
        ):
            _numba_master_vectorized(
                *args_flags,
                out=tuple(self.work_arrays),
            )
            if self.constraint_intensity:
                penalty, dpenalty, dpenalty_binding = self.constraint_penalty()
                self.work_arrays.loglike[:] += penalty
                self.work_arrays.d_loglike[:] += np.expand_dims(dpenalty_binding, 0)
                self.work_arrays.bhhh[:] = np.einsum(
                    "ij,ik->ijk", self.work_arrays.d_loglike, self.work_arrays.d_loglike
                )
        bhhh = self.work_arrays.bhhh.sum(0)
        dloglike = self.work_arrays.d_loglike.sum(0)
        freedoms = self.pholdfast == 0
        from .numba_optimization import propose_direction

        direction = propose_direction(bhhh, dloglike, freedoms)
        tolerance = np.dot(direction, dloglike) - self.n_cases
        return tolerance

    def _loglike_runner(
        self,
        x=None,
        only_utility=0,
        return_gradient=False,
        return_probability=False,
        return_bhhh=False,
        start_case=None,
        stop_case=None,
        step_case=None,
    ):
        caseslice = slice(start_case, stop_case, step_case)
        args = self.__prepare_for_compute(
            x,
            allow_missing_ch=return_probability or (only_utility > 0),
            caseslice=caseslice,
        )
        args_flags = args + (
            np.asarray(
                [
                    only_utility,
                    return_probability,
                    return_gradient,
                    return_bhhh,
                ],
                dtype=np.int8,
            ),
        )
        try:
            with np.errstate(
                divide="ignore",
                over="ignore",
            ):
                try:
                    result_arrays = WorkArrays(
                        *_numba_master_vectorized(
                            *args_flags,
                            out=tuple(self.work_arrays.cs[caseslice]),
                        )
                    )
                except ValueError:
                    result_arrays = WorkArrays(
                        *_numba_master_vectorized(
                            *args_flags,
                            # out=tuple(self.work_arrays.cs[caseslice]),
                        )
                    )

                if self.constraint_intensity:
                    penalty, dpenalty, dpenalty_binding = self.constraint_penalty()
                    self.work_arrays.loglike[caseslice] += np.nan_to_num(penalty)
                    self.work_arrays.d_loglike[caseslice] += np.nan_to_num(
                        np.expand_dims(dpenalty, 0)
                    )
                    self.work_arrays.bhhh[caseslice] = np.nan_to_num(
                        np.einsum(
                            "ij,ik->ijk",
                            self.work_arrays.d_loglike[caseslice],
                            self.work_arrays.d_loglike[caseslice],
                        )
                    )
                else:
                    penalty = 0.0

        except Exception:
            shp = lambda y: getattr(y, "shape", "scalar")
            dtp = lambda y: getattr(y, "dtype", f"{type(y)} ")
            import inspect

            arg_names = list(inspect.signature(_numba_master).parameters)
            arg_name_width = max(len(j) for j in arg_names)

            in_sig, out_sig = _master_shape_signature.split("->")
            in_sig_shapes = in_sig.split("(")[1:]
            out_sig_shapes = out_sig.split("(")[1:]
            print(in_sig_shapes)
            print(out_sig_shapes)
            print("# Input Arrays")
            for n, (a, s) in enumerate(zip(args_flags, in_sig_shapes)):
                s = s.rstrip(" ),")
                print(
                    f" {arg_names[n]:{arg_name_width}} [{n:2}] {s.strip():9}: {dtp(a)}{shp(a)}"
                )
            print("# Output Arrays")
            final_n = n
            for n, (a, s) in enumerate(
                zip(self.work_arrays, out_sig_shapes), start=final_n + 1
            ):
                s = s.rstrip(" ),")
                print(
                    f" {arg_names[n]:{arg_name_width}} [{n:2}] {s.strip():9}: {dtp(a)}{shp(a)}"
                )
            raise
        return result_arrays, penalty

    @property
    def weight_normalization(self):
        # try:
        #     return self.dataframes.weight_normalization
        # except AttributeError:
        return 1.0

    @weight_normalization.setter
    def weight_normalization(self, x):
        pass  # TODO

    def loglike(
        self,
        x=None,
        *,
        start_case: int | None = None,
        stop_case: int | None = None,
        step_case: int | None = None,
        check_if_best: bool = True,
        error_if_bad: bool = True,
        **kwargs,
    ):
        """
        Compute the log likelihood of the model.

        Parameters
        ----------
        x : array-like or dict, optional
            New values to set for the parameters before evaluating
            the log likelihood.  If given as array-like, the array must
            be a vector with length equal to the length of the
            parameter frame, and the given vector will replace
            the current values.  If given as a dictionary,
            the dictionary is used to update the parameters.
        start_case : int, default 0
            The first case to include in the log likelihood computation.
            To include all cases, start from 0 (the default).
        stop_case : int, default -1
            One past the last case to include in the log likelihood
            computation.  This is processed as usual for Python slicing
            and iterating, and negative values count backward from the
            end.  To include all cases, end at -1 (the default).
        step_case : int, default 1
            The step size of the case iterator to use in likelihood
            calculation.  This is processed as usual for Python slicing
            and iterating.  To include all cases, step by 1 (the default).
        check_if_best : bool, default True
            If True, check if the current log likelihood is the best
            found so far, and if so, update the cached best log likelihood
            and cached best parameters.
        error_if_bad : bool, default True
            If True, raise an exception if the log likelihood is NaN or Inf.

        Returns
        -------
        float
        """
        if self._use_streaming:
            result = (
                self.streaming.loglike(
                    x, start_case=start_case, stop_case=stop_case, step_case=step_case
                )
                * self.weight_normalization
            )
        else:
            result_arrays, penalty = self._loglike_runner(
                x, start_case=start_case, stop_case=stop_case, step_case=step_case
            )
            result = result_arrays.loglike.sum() * self.weight_normalization
            if error_if_bad:
                for f, tag in [(np.isnan, "NaN"), (np.isinf, "Inf")]:
                    if f(result):
                        if f(self.weight_normalization):
                            raise ValueError(f"weight_normalization is {tag}")
                        msg = f"log likelihood is {tag}"
                        bad_case_indexes = np.where(f(result_arrays.loglike))[0]
                        if len(bad_case_indexes) > 0:
                            msg += (
                                f" in {len(bad_case_indexes)} cases, including CASEIDs:"
                            )
                            caseids = self.dataset.dc.caseids()
                            msg += f" {caseids[bad_case_indexes[0]]}"
                            for i in bad_case_indexes[1:5]:
                                msg += f", {caseids[i]}"
                            if len(bad_case_indexes) > 5:
                                msg += ", ..."
                        raise ValueError(msg)
        if start_case is None and stop_case is None and step_case is None:
            self._check_if_best(result)
        return result

    def d_loglike(
        self,
        x=None,
        *,
        start_case=None,
        stop_case=None,
        step_case=None,
        return_series=False,
        **kwargs,
    ):
        if self._use_streaming:
            result = (
                self.streaming.d_loglike(
                    x, start_case=start_case, stop_case=stop_case, step_case=step_case
                )
                * self.weight_normalization
            )
        else:
            result_arrays, penalty = self._loglike_runner(
                x,
                start_case=start_case,
                stop_case=stop_case,
                step_case=step_case,
                return_gradient=True,
            )
            result = result_arrays.d_loglike.sum(0) * self.weight_normalization
        if return_series:
            result = pd.Series(result, index=self.pnames)
        return result

    def loglike_casewise(
        self,
        x=None,
        *,
        start_case=None,
        stop_case=None,
        step_case=None,
        **kwargs,
    ):
        result_arrays, penalty = self._loglike_runner(
            x,
            start_case=start_case,
            stop_case=stop_case,
            step_case=step_case,
        )
        return result_arrays.loglike * self.weight_normalization

    def loglike_problems(self) -> pd.DataFrame:
        """
        Identify cases with log likelihood problems.

        Returns
        -------
        pandas.DataFrame
            A DataFrame identifying the caseids of cases with log likelihood
            problems, and the nature of the problem (e.g. NaN, +Inf, -Inf,
            exactly zero).
        """
        llcase = self.loglike_casewise()
        caseids = self.datatree.caseids()
        problems = pd.Series(np.nan, index=caseids, name="problem")
        problems.iloc[np.where(np.isnan(llcase))] = "nan"
        problems.iloc[np.where(np.isposinf(llcase))] = "+inf"
        problems.iloc[np.where(np.isneginf(llcase))] = "-inf"
        problems.iloc[np.where(llcase == 0)] = "zero"
        return problems.reset_index().rename_axis("caseindex").dropna()

    def loglike_null(self, use_cache=True):
        """
        Compute the log likelihood at null values.

        Set all parameter values to the value indicated in the
        "nullvalue" column of the parameter frame, and compute
        the log likelihood with the currently loaded data.  Note
        that the null value for each parameter may not be zero
        (for example, the default null value for logsum parameters
        in a nested logit model is 1).

        Parameters
        ----------
        use_cache : bool, default True
            Use the cached value if available.  Set to -1 to
            raise an exception if there is no cached value.

        Returns
        -------
        float
        """
        if self._cached_loglike_null is not None and use_cache:
            return self._cached_loglike_null
        elif use_cache == -1:
            raise ValueError("no cached value")
        else:
            current_parameters = self.pvals.copy()
            self.pvals = "null"
            self._cached_loglike_null = self.loglike()
            self.pvals = current_parameters
            return self._cached_loglike_null

    def d_loglike_casewise(
        self,
        x=None,
        *,
        start_case=None,
        stop_case=None,
        step_case=None,
        **kwargs,
    ):
        result_arrays, penalty = self._loglike_runner(
            x,
            start_case=start_case,
            stop_case=stop_case,
            step_case=step_case,
            return_gradient=True,
        )
        return result_arrays.d_loglike * self.weight_normalization

    def bhhh(
        self,
        x=None,
        *,
        start_case=None,
        stop_case=None,
        step_case=None,
        return_dataframe=False,
        **kwargs,
    ):
        result_arrays, penalty = self._loglike_runner(
            x,
            start_case=start_case,
            stop_case=stop_case,
            step_case=step_case,
            return_bhhh=True,
        )
        result = result_arrays.bhhh.sum(0) * self.weight_normalization
        if return_dataframe:
            result = pd.DataFrame(result, columns=self.pnames, index=self.pnames)
        return result

    def robust_covariance(self) -> xr.DataArray:
        """
        Compute the robust covariance matrix of the parameter estimates.

        Returns
        -------
        xarray.DataArray
        """
        if "robust_covariance_matrix" in self.parameters:
            return self.parameters["robust_covariance_matrix"]
        cov_dataarray = self.parameters["covariance_matrix"]
        covariance_matrix = cov_dataarray.to_numpy()
        take = np.full_like(covariance_matrix, True, dtype=bool)
        dense_s = len(self.pvals)
        for i in range(dense_s):
            if self.pholdfast[i]:
                # or (self.pf.loc[ii, 'value'] >= self.pf.loc[ii, 'maximum'])
                # or (self.pf.loc[ii, 'value'] <= self.pf.loc[ii, 'minimum']):
                take[i, :] = False
                take[:, i] = False
                dense_s -= 1
        if dense_s > 0:
            covariance_taken = covariance_matrix[take].reshape(dense_s, dense_s)
            # try:
            #     invhess = np.linalg.inv(hess_taken)
            # except np.linalg.linalg.LinAlgError:
            #     invhess = np.full_like(hess_taken, np.nan, dtype=np.float64)
            # robust...
            try:
                bhhh_taken = self.bhhh()[take].reshape(dense_s, dense_s)
            except NotImplementedError:
                robust_covariance_matrix = np.full_like(
                    covariance_matrix, 0, dtype=np.float64
                )
            else:
                # import scipy.linalg.blas
                # temp_b_times_h = scipy.linalg.blas.dsymm(float(1), invhess, bhhh_taken)
                # robusto = scipy.linalg.blas.dsymm(float(1), invhess, temp_b_times_h, side=1)
                robusto = np.dot(np.dot(covariance_taken, bhhh_taken), covariance_taken)
                robust_covariance_matrix = np.full_like(
                    covariance_matrix, 0, dtype=np.float64
                )
                robust_covariance_matrix[take] = robusto.reshape(-1)
        else:
            robust_covariance_matrix = np.full_like(
                covariance_matrix, 0, dtype=np.float64
            )

        self.add_parameter_array("robust_covariance_matrix", robust_covariance_matrix)
        self.add_parameter_array(
            "robust_std_err", _safe_sqrt(np.diagonal(robust_covariance_matrix))
        )
        return self.parameters["robust_covariance_matrix"]

    def _wrap_as_dataframe(
        self,
        arr,
        return_type: Literal["dataframe", "names", "idce", "idca", "dataarray", None],
        start_case: int | None = None,
        stop_case: int | None = None,
        step_case: int | None = None,
    ):
        """
        Wrap the given array as a DataFrame or DataArray.

        Parameters
        ----------
        arr : array-like
            The array to wrap.
        return_type : {"dataframe", "names", "idca", "dataarray", None}, default None
            Return the result in the indicated format.
            - 'dataframe' gives a pandas DataFrame indexed by cases and with
                alternative codes as columns.
            - 'names' gives a pandas DataFrame indexed by cases and with
                alternative names as columns.
            - 'idca' gives a pandas DataFrame containing a single columns and
                with a two-level multi-index giving cases and alternatives.
            - 'dataarray' gives a two-dimension DataArray, with cases and
                alternatives as dimensions.
        """
        if return_type:
            idx = self.datatree.caseids()
            if idx is not None:
                idx = idx[start_case:stop_case:step_case]
            if return_type == "names":
                return pd.DataFrame(
                    data=arr,
                    columns=pd.Index(
                        self.graph.standard_sort_names[: arr.shape[1]], name="alt_name"
                    ),
                    index=idx,
                )
            if return_type == "dataarray":
                return DataArray(
                    arr,
                    coords={
                        self.datatree.CASEID: idx,
                        "nodeid": np.asarray(self.graph.standard_sort),
                        "node_name": DataArray(
                            np.asarray(self.graph.standard_sort_names), dims="nodeid"
                        ),
                    },
                    dims=(self.datatree.CASEID, "nodeid"),
                )
            result = pd.DataFrame(
                data=arr,
                columns=pd.Index(
                    self.graph.standard_sort[: arr.shape[1]],
                    name=self.datatree.dc.ALTID,
                ),
                index=idx,
            )
            if return_type == "idce":
                raise NotImplementedError
            elif return_type == "idca":
                return result.stack()
            else:
                return result
        return arr

    def probability(
        self,
        x=None,
        *,
        start_case: int | None = None,
        stop_case: int | None = None,
        step_case: int | None = None,
        return_format: Literal[
            "dataframe", "names", "idce", "idca", "dataarray", None
        ] = False,
        include_nests=False,
    ):
        """
        Compute values for the probability function embodied by the model.

        Parameters
        ----------
        x : array-like or dict, optional
            New values to set for the parameters before evaluating
            the probability.  If given as array-like, the array must
            be a vector with length equal to the length of the
            parameter frame, and the given vector will replace
            the current values.  If given as a dictionary,
            the dictionary is used to update the parameters.
        start_case : int, default 0
            The first case to include in the probability computation.
            To include all cases, start from 0 (the default).
        stop_case : int, default -1
            One past the last case to include in the probability
            computation.  This is processed as usual for Python slicing
            and iterating, and negative values count backward from the
            end.  To include all cases, end at -1 (the default).
        step_case : int, default 1
            The step size of the case iterator to use in probability
            calculation.  This is processed as usual for Python slicing
            and iterating.  To include all cases, step by 1 (the default).
        return_format : {"dataframe", "names", "idca", "dataarray", None}, default None
            Return the result in the indicated format.
            - 'dataframe' gives a pandas DataFrame indexed by cases and with
                alternative codes as columns.
            - 'names' gives a pandas DataFrame indexed by cases and with
                alternative names as columns.
            - 'idca' gives a pandas DataFrame containing a single columns and
                with a two-level multi-index giving cases and alternatives.
            - 'dataarray' gives a two-dimension DataArray, with cases and
                alternatives as dimensions.

        Returns
        -------
        array or DataFrame or DataArray
        """
        result_arrays, penalty = self._loglike_runner(
            x,
            start_case=start_case,
            stop_case=stop_case,
            step_case=step_case,
            return_probability=True,
        )
        pr = result_arrays.probability
        if not include_nests:
            pr = pr[:, : self.graph.n_elementals()]
        return self._wrap_as_dataframe(
            pr,
            return_format,
            start_case=start_case,
            stop_case=stop_case,
            step_case=step_case,
        )

    def utility(
        self,
        x=None,
        *,
        start_case=None,
        stop_case=None,
        step_case=None,
        return_format=None,
    ):
        """
        Compute values for the utility function contained in the model.

        Parameters
        ----------
        x : array-like or dict, optional
            New values to set for the parameters before evaluating
            the log likelihood.  If given as array-like, the array must
            be a vector with length equal to the length of the
            parameter frame, and the given vector will replace
            the current values.  If given as a dictionary,
            the dictionary is used to update the parameters.
        start_case : int, default 0
            The first case to include in the log likelihood computation.
            To include all cases, start from 0 (the default).
        stop_case : int, default -1
            One past the last case to include in the log likelihood
            computation.  This is processed as usual for Python slicing
            and iterating, and negative values count backward from the
            end.  To include all cases, end at -1 (the default).
        step_case : int, default 1
            The step size of the case iterator to use in likelihood
            calculation.  This is processed as usual for Python slicing
            and iterating.  To include all cases, step by 1 (the default).
        return_format : {None, 'idco', 'dataarray'}, default None
            Return the result in the indicated format.
              - 'idco' gives a pandas DataFrame indexed by cases and with
                alternatives as columns.
              - 'idca' gives a pandas Series with a two-level multi-index.
              - 'dataarray' gives a two-dimension larch DataArray.

        Returns
        -------
        array or DataFrame or DataArray
        """
        result_arrays, penalty = self._loglike_runner(
            x,
            start_case=start_case,
            stop_case=stop_case,
            step_case=step_case,
            only_utility=2,
        )
        return self._wrap_as_dataframe(
            result_arrays.utility,
            return_format,
            start_case=start_case,
            stop_case=stop_case,
            step_case=step_case,
        )

    def quantity(
        self,
        x=None,
        *,
        start_case=None,
        stop_case=None,
        step_case=None,
        return_type: Literal[
            "dataframe", "names", "idce", "idca", "dataarray", None
        ] = None,
    ):
        result_arrays, penalty = self._loglike_runner(
            x,
            start_case=start_case,
            stop_case=stop_case,
            step_case=step_case,
            only_utility=3,
        )
        return self._wrap_as_dataframe(
            result_arrays.utility,
            return_type,
        )

    def logsums(
        self,
        x=None,
        *,
        start_case=None,
        stop_case=None,
        step_case=None,
        arr=None,
    ):
        result_arrays, penalty = self._loglike_runner(
            x,
            start_case=start_case,
            stop_case=stop_case,
            step_case=step_case,
            only_utility=2,
        )
        if arr is not None:
            arr[start_case:stop_case:step_case] = result_arrays.utility[:, -1]
            return arr
        return result_arrays.utility[:, -1]

    def loglike2(
        self,
        x=None,
        *,
        start_case=None,
        stop_case=None,
        step_case=None,
        persist=0,
        return_series=True,
        probability_only=False,
    ):
        result_arrays, penalty = self._loglike_runner(
            x,
            start_case=start_case,
            stop_case=stop_case,
            step_case=step_case,
            return_gradient=True,
        )
        result = dictx(
            ll=result_arrays.loglike.sum() * self.weight_normalization,
            dll=result_arrays.d_loglike.sum(0) * self.weight_normalization,
        )
        if start_case is None and stop_case is None and step_case is None:
            self._check_if_best(result.ll)
        if return_series:
            result["dll"] = pd.Series(
                result["dll"],
                index=self.pnames,
            )
        return result

    def loglike2_bhhh(
        self,
        x=None,
        *,
        return_series=False,
        start_case=None,
        stop_case=None,
        step_case=None,
        persist=0,
    ):
        result_arrays, penalty = self._loglike_runner(
            x,
            start_case=start_case,
            stop_case=stop_case,
            step_case=step_case,
            return_gradient=True,
            return_bhhh=True,
        )
        result = dictx(
            ll=result_arrays.loglike.sum() * self.weight_normalization,
            dll=result_arrays.d_loglike.sum(0) * self.weight_normalization,
            bhhh=result_arrays.bhhh.sum(0) * self.weight_normalization,
        )
        from ..model.persist_flags import (
            PERSIST_D_LOGLIKE_CASEWISE,
            PERSIST_LOGLIKE_CASEWISE,
        )

        if persist & PERSIST_LOGLIKE_CASEWISE:
            result["ll_casewise"] = result_arrays.loglike * self.weight_normalization
        if persist & PERSIST_D_LOGLIKE_CASEWISE:
            result["dll_casewise"] = result_arrays.d_loglike * self.weight_normalization
        if start_case is None and stop_case is None and step_case is None:
            self._check_if_best(result.ll)
        if return_series:
            result["dll"] = pd.Series(
                result["dll"],
                index=self.pnames,
            )
            result["bhhh"] = pd.DataFrame(
                result["bhhh"], index=self.pnames, columns=self.pnames
            )
        result["penalty"] = penalty
        return result

    def d2_loglike(
        self,
        x=None,
        *,
        start_case=None,
        stop_case=None,
        step_case=None,
    ):
        if x is None:
            x = self.pvals.copy()
        from ..util.math import approx_fprime

        result = approx_fprime(
            x,
            lambda y: self.d_loglike(
                y,
                start_case=start_case,
                stop_case=stop_case,
                step_case=step_case,
            ),
        )
        # the approx_fprime function will leave a residual epsilon on the last
        # parameter, we need to clean that up and restore the parameters to their
        # original values
        self.pvals = x
        return result

    def neg_loglike(
        self,
        x=None,
        start_case=None,
        stop_case=None,
        step_case=None,
    ):
        result = self.loglike(
            x,
            start_case=start_case,
            stop_case=stop_case,
            step_case=step_case,
        )
        return -result

    def logloss(
        self,
        x=None,
        *,
        start_case=None,
        stop_case=None,
        step_case=None,
        check_if_best=True,
    ):
        result = self.loglike(
            x,
            start_case=start_case,
            stop_case=stop_case,
            step_case=step_case,
            check_if_best=check_if_best,
        )
        return -result / self.total_weight()

    def neg_d_loglike(self, x=None, start_case=0, stop_case=-1, step_case=1, **kwargs):
        result = self.d_loglike(
            x, start_case=start_case, stop_case=stop_case, step_case=step_case, **kwargs
        )
        return -np.asarray(result)

    def d_logloss(self, x=None, start_case=0, stop_case=-1, step_case=1, **kwargs):
        result = self.d_loglike(
            x, start_case=start_case, stop_case=stop_case, step_case=step_case, **kwargs
        )
        return -np.asarray(result) / self.total_weight()

    def jumpstart_bhhh(
        self,
        steplen=0.5,
        jumpstart=0,
        jumpstart_split=5,
        logger=None,
    ):
        """
        Jump start optimization.

        Parameters
        ----------
        steplen
        jumpstart
        jumpstart_split

        """
        if logger is None:

            class NoLogger:
                debug = lambda *x: None
                info = lambda *x: None

            logger = NoLogger()

        for _jump in range(jumpstart):
            j_pvals = self.pvals.copy()
            #
            # jump_breaks = list(
            #     range(0, n_cases, n_cases // jumpstart_split + (1 if n_cases % jumpstart_split else 0))
            # ) + [n_cases]
            #
            # for j0, j1 in zip(jump_breaks[:-1], jump_breaks[1:]):
            for j0 in range(jumpstart_split):
                result = self.loglike2_bhhh(
                    start_case=j0,
                    step_case=jumpstart_split,
                )
                current_dll = result.dll
                current_bhhh = result.bhhh
                bhhh_inv = self._free_slots_inverse_matrix(current_bhhh)
                direction = np.dot(current_dll, bhhh_inv)
                j_pvals += direction * steplen
                logger.debug(f"jump to {j_pvals}")
                self.set_values(j_pvals)

    def check_d_loglike(self, stylize=True, skip_zeros=False):
        """
        Check that the analytic and finite-difference gradients are approximately equal.

        Primarily used for debugging.

        Parameters
        ----------
        stylize : bool, default True
            See :ref:`check_gradient` for details.
        skip_zeros : bool, default False
            See :ref:`check_gradient` for details.

        Returns
        -------
        pd.DataFrame or Stylized DataFrame
        """
        epsilon = np.sqrt(np.finfo(self.float_dtype).eps)
        from .gradient_check import check_gradient

        return check_gradient(
            self.loglike,
            self.d_loglike,
            self.pvals.copy(),
            names=list(self.pnames),
            stylize=stylize,
            skip_zeros=skip_zeros,
            epsilon=epsilon,
        )

    def __getstate__(self):
        attr = {}
        port = [
            "availability_any",
            "availability_ca_var",
            "availability_co_vars",
            "choice_any",
            "choice_ca_var",
            "choice_co_code",
            "choice_co_vars",
            "common_draws",
            "_compute_engine",
            "constraint_intensity",
            "constraint_sharpness",
            "constraints",
            # "dataflows",
            # 'dataset',
            # 'datatree',
            "float_dtype",
            "graph",
            "groupid",
            "logsum_parameter",
            "mixtures",
            "n_draws",
            "parameters",
            "prerolled_draws",
            "quantity_ca",
            "quantity_scale",
            "rename_parameters",
            "seed",
            "title",
            "utility_ca",
            "utility_co",
            "weight_co_var",
        ]
        for k in port:
            attr[k] = getattr(self, k)
        return attr

    def __setstate__(self, state):
        defer = {
            "constraints",
        }
        for check in [lambda i: i in defer, lambda i: i not in defer]:
            for k, v in state.items():
                if check(k):
                    continue
                if k == "parameters":
                    self._parameter_bucket.update_parameters(v)
                else:
                    try:
                        setattr(self, k, v)
                    except AttributeError as err:
                        raise AttributeError(f"{k}: {err}") from None

    def copy(self, datatree=True):
        dupe = type(self)()
        dupe.__setstate__(self.__getstate__())
        if datatree:
            dupe.datatree = self.datatree
        return dupe

    def remove_unused_parameters(self, verbose=True):
        """
        Remove parameters that are not used in the model.

        Parameters
        ----------
        verbose : bool, default True
            Generate log messages about how many parameters were dropped.
        """
        old_param_names = self.pnames
        old_parameters = self.parameters
        # clear existing parameters
        self._parameter_bucket._params = old_parameters.reindex(
            {self._parameter_bucket.index_name: []}
        )
        self.unmangle(True)
        new_param_names = self.pnames
        self._parameter_bucket._params = old_parameters.reindex(
            {self._parameter_bucket.index_name: new_param_names}
        )
        if verbose:
            dropped_params = [p for p in old_param_names if p not in new_param_names]
            if len(dropped_params) == 0:
                pass
            elif len(dropped_params) < 4:
                logging.getLogger("Larch").warning(
                    f"dropped {len(dropped_params)} parameters: {', '.join(dropped_params)}"
                )
            else:
                logging.getLogger("Larch").warning(
                    f"dropped {len(dropped_params)} parameters including: "
                    f"{', '.join(dropped_params[:3])}"
                )

    @property
    def n_cases(self) -> int:
        """The number of cases in the attached data."""
        data_as_possible = self.data_as_possible
        if data_as_possible is None:
            raise MissingDataError("no data are set")
        return data_as_possible.n_cases

    def total_weight(self) -> float:
        """
        Compute the total weight of cases in the loaded data.

        Returns
        -------
        float
        """
        if self._data_arrays is not None:
            if self._data_arrays.ch.ndim - 1 == self._data_arrays.wt.ndim:
                return (self._data_arrays.ch[:, -1] * self._data_arrays.wt).sum()
            elif self._data_arrays.ch.ndim - 2 == self._data_arrays.wt.ndim:
                return (
                    self._data_arrays.ch[:, -1]
                    * np.expand_dims(self._data_arrays.wt, -1)
                ).sum()
            elif self._data_arrays.ch.ndim - 3 == self._data_arrays.wt.ndim:
                return (
                    self._data_arrays.ch[:, -1]
                    * np.expand_dims(self._data_arrays.wt, (-1, -2))
                ).sum()
            else:
                raise ValueError
        if self.use_streaming:
            return self.streaming.total_weight()
        raise MissingDataError("no data_arrays are set")

    @property
    def dataset(self) -> Dataset | None:
        """xarray.Dataset : Data arrays as loaded for model computation."""
        super().unmangle()
        if self._dataset is None:
            self.reflow_data_arrays()
        try:
            return self._dataset
        except AttributeError:
            return None

    @dataset.setter
    def dataset(self, dataset: Dataset):
        if dataset is self._dataset:
            return
        if isinstance(dataset, Dataset):
            self._dataset = dataset
            self._data_arrays = None
        else:
            raise TypeError(f"dataset must be Dataset not {type(dataset)}")

    @dataset.deleter
    def dataset(self):
        self._dataset = None
        self._data_arrays = None

    @property
    def data_as_loaded(self):
        if self.dataset is not None:
            return self.dataset
        return None

    @property
    def data_as_possible(self):
        if self._dataset is not None:
            return self._dataset.dc
        if self.datatree is not None:
            return self.datatree
        return None

    @property
    def float_dtype(self):
        try:
            return self._float_dtype
        except AttributeError:
            return None

    @float_dtype.setter
    def float_dtype(self, float_dtype):
        if self.float_dtype != float_dtype:
            self.mangle()
        self._float_dtype = float_dtype

    def choice_avail_summary(self):
        """
        Generate a summary of choice and availability statistics.

        Returns
        -------
        pandas.DataFrame
        """
        from ..dataset.choice_avail_reporting import choice_avail_summary

        self.unmangle()
        graph = self.graph
        if self.use_streaming:
            return self.streaming.choice_avail_summary()
            # from .numba_stream import init_choice_avail_summary_streamer
            # choice_avail_summary_streamer = init_choice_avail_summary_streamer(self)
            # total_ch, total_av, total_wt = choice_avail_summary_streamer(
            #     self.n_cases,
            #     self._fixed_arrays.edge_slots,
            #     self._fixed_arrays.mu_slot,
            #     self.datatree.n_alts,
            # )
            # od = {}
            # idx = pd.Index(graph.standard_sort, name="altid")
            # od["name"] = pd.Series(graph.standard_sort_names, index=idx)
            # od["chosen"] = pd.Series(total_ch, index=idx)
            # od["available"] = pd.Series(total_av, index=idx)
            # result = pd.DataFrame(od,  index = idx)
            # from ..dataset.choice_avail_reporting import clean_summary
            # return clean_summary(result, root_id=graph.root_id)
        else:
            return choice_avail_summary(
                self.dataset,
                graph,
                self.availability_co_vars,
            )

    def maximize_loglike(
        self,
        *args,
        **kwargs,
    ) -> dictx:
        """
        Maximize the log likelihood.

        Parameters
        ----------
        method : str, optional
            The optimization method to use.  See scipy.optimize for
            most possibilities, or use 'BHHH'. Defaults to SLSQP if
            there are any constraints or finite parameter bounds,
            otherwise defaults to BHHH.
        quiet : bool, default False
            Whether to suppress the dashboard.
        options : dict, optional
            These options are passed through to the `scipy.optimize.minimize`
            function.
        maxiter : int, optional
            Maximum number of iterations.  This argument is just added to
            `options` for most methods.

        Returns
        -------
        larch.util.dictx
            A dictionary of results, including final log likelihood,
            elapsed time, and other statistics.  The exact items
            included in output will vary by estimation method.

        """
        from .optimization import maximize_loglike

        return maximize_loglike(self, *args, **kwargs)

    def _get_bounds_constraints(self, binding_tol=1e-4):
        """
        Convert bounds to parametric constraints on the model.

        Parameters
        ----------
        binding_tol : Number or Mapping
            The binding tolerance to use, which determines
            whether a constraint is considered active or not.

        Returns
        -------
        list
            A list of constraints.
        """
        from collections.abc import Mapping
        from numbers import Number

        from .constraints import FixedBound

        if isinstance(binding_tol, Number):
            default_binding_tol = binding_tol
        else:
            default_binding_tol = 1e-4
        if not isinstance(binding_tol, Mapping):
            binding_tol = {}
        get_bind_tol = lambda x: binding_tol.get(x, default_binding_tol)
        bounds = []
        self.unmangle()
        for pname, phold, pmin, pmax in zip(
            self.pnames,
            self.pholdfast,
            self.pminimum,
            self.pmaximum,
        ):
            if phold:
                # don't create bounds constraints on holdfast parameters
                continue
            if pmin != -np.inf or pmax != np.inf:
                bounds.append(
                    FixedBound(
                        pname, pmin, pmax, model=self, binding_tol=get_bind_tol(pname)
                    )
                )
        return bounds

    def _get_constraints(self, method):
        if method.lower() in ("slsqp", "cobyla"):
            constraint_dicts = []
            for c in self.constraints:
                constraint_dicts.extend(c.as_constraint_dicts())
            return constraint_dicts
        if method.lower() in ("trust-constr"):
            constraints = []
            for c in self.constraints:
                constraints.extend(c.as_linear_constraints())
            return constraints
        return ()

    def check_for_overspecification(self, pvals=None):
        """
        Check model for possible over-specification.

        Parameters
        ----------
        pvals : array-like, optional
            The parameter values to use in the calculation.  If not
            given, the current parameter values are used.

        Returns
        -------
        list of tuples
            A list of possible overspecification problems in the model.  Each problem
            is a tuple containing the eigenvalue, the indices of the non-zero elements
            in the eigenvector, and the eigenvector itself.
        """
        if pvals is None:
            pvals = self.pvals
        locks = np.asarray(self.pholdfast.astype(bool))
        if self.compute_engine == "jax":
            _se, hess, _inv_hess = self.jax_param_cov(pvals)
        else:
            hess = -self.d2_loglike(pvals)
        hess = np.asarray(hess).copy()
        hess[locks, :] = 0
        hess[:, locks] = 0

        overspec = compute_possible_overspecification(hess, self.pholdfast)
        if overspec:
            possible_overspecification = []
            msg = "Model is possibly over-specified (hessian is nearly singular)."
            msg += "\nLook for problems in these parameters or groups of parameters:"
            for eigval, ox, eigenvec in overspec:
                if eigval == "LinAlgError":
                    possible_overspecification.append((eigval, [ox], [""]))
                else:
                    paramset = list(np.asarray(self.pnames)[ox])
                    possible_overspecification.append((eigval, paramset, eigenvec[ox]))
                    msg += f"\n- Eigenvalue: {eigval}"
                    max_len_param = max(len(p) for p in paramset)
                    for p, z in zip(paramset, eigenvec[ox]):
                        msg += f"\n    {p:{max_len_param}s}: {z}"
            self._possible_overspecification = possible_overspecification
            warnings.warn(
                msg,
                category=PossibleOverspecification,
                stacklevel=2,
            )
            return possible_overspecification

    def calculate_parameter_covariance(self, pvals=None, *, robust=False):
        """
        Calculate the parameter covariance matrix.

        Parameters
        ----------
        pvals : array-like, optional
            The parameter values to use in the calculation.  If not
            given, the current parameter values are used.
        robust : bool, default False
            Whether to calculate the robust covariance matrix.

        Returns
        -------
        se : array
            The standard errors of the parameter estimates.
        hess : array
            The Hessian matrix.
        ihess : array
            The inverse of the Hessian matrix.
        """
        if pvals is None:
            pvals = self.pvals
        locks = np.asarray(self.pholdfast.astype(bool))
        if self.compute_engine == "jax":
            se, hess, ihess = self.jax_param_cov(pvals)
        else:
            hess = -self.d2_loglike(pvals)
            if self.parameters["holdfast"].sum():
                free = self.pholdfast == 0
                hess_ = hess[free][:, free]
                try:
                    ihess_ = np.linalg.inv(hess_)
                except np.linalg.LinAlgError:
                    ihess_ = np.linalg.pinv(hess_)
                ihess = _arr_inflate(ihess_, locks)
            else:
                try:
                    ihess = np.linalg.inv(hess)
                except np.linalg.LinAlgError:
                    ihess = np.linalg.pinv(hess)
            se = _safe_sqrt(ihess.diagonal())
            self.pstderr = se
        hess = np.asarray(hess).copy()
        hess[locks, :] = 0
        hess[:, locks] = 0
        ihess = np.asarray(ihess).copy()
        ihess[locks, :] = 0
        ihess[:, locks] = 0
        self.add_parameter_array("hess", hess)
        self.add_parameter_array("ihess", ihess)

        overspec = compute_possible_overspecification(hess, self.pholdfast)
        if overspec:
            warnings.warn(
                "Model is possibly over-specified (hessian is nearly singular).",
                category=PossibleOverspecification,
                stacklevel=2,
            )
            possible_overspecification = []
            for eigval, ox, eigenvec in overspec:
                if eigval == "LinAlgError":
                    possible_overspecification.append((eigval, [ox], [""]))
                else:
                    paramset = list(np.asarray(self.pnames)[ox])
                    possible_overspecification.append((eigval, paramset, eigenvec[ox]))
            self._possible_overspecification = possible_overspecification

        # constrained covariance
        if self.constraints:
            constraints = list(self.constraints)
        else:
            constraints = []
        try:
            constraints.extend(self._get_bounds_constraints())
        except Exception:
            pass

        if constraints:
            binding_constraints = list()
            self.add_parameter_array("unconstrained_std_err", self.pstderr)
            self.add_parameter_array("unconstrained_covariance_matrix", ihess)

            s = np.asarray(ihess)
            pvals = self.pvals
            for c in constraints:
                if np.absolute(c.fun(pvals)) < c.binding_tol:
                    binding_constraints.append(c)
                    b = c.jac(self.pf.value)
                    den = b @ s @ b
                    if den != 0:
                        s = s - (1 / den) * s @ b.reshape(-1, 1) @ b.reshape(1, -1) @ s
            self.add_parameter_array("covariance_matrix", s)
            self.pstderr = _safe_sqrt(s.diagonal())

            # Fix numerical issues on some constraints, add constrained notes
            if binding_constraints or any(self.pholdfast != 0):
                notes = {}
                for c in binding_constraints:
                    pa = c.get_parameters()
                    for p in pa:
                        # if self.pf.loc[p, 't_stat'] > 1e5:
                        #     self.pf.loc[p, 't_stat'] = np.inf
                        #     self.pf.loc[p, 'std_err'] = np.nan
                        # if self.pf.loc[p, 't_stat'] < -1e5:
                        #     self.pf.loc[p, 't_stat'] = -np.inf
                        #     self.pf.loc[p, 'std_err'] = np.nan
                        n = notes.get(p, [])
                        n.append(c.get_binding_note(pvals))
                        notes[p] = n
                constrained_note = (
                    pd.Series({k: "\n".join(v) for k, v in notes.items()}, dtype=object)
                    .reindex(self.pnames)
                    .fillna("")
                )
                constrained_note[self.pholdfast != 0] = "fixed value"
                self.add_parameter_array("constrained", constrained_note)

        if robust:
            self.robust_covariance()
            se = self.parameters["robust_std_err"]

        return se, hess, ihess

    def histogram_on_idca_variable(
        self,
        x,
        **kwargs,
    ):
        from ..util.figures import histogram_on_idca_variable

        return histogram_on_idca_variable(
            x,
            pr=self.probability(),  # array-like
            ds=self.dataset,
            dt=self.datatree,
            **kwargs,
        )

    def utility_breakdown(
        self,
        altid: int,
        *,
        caseid: int | None = None,
        caseindex: int | None = None,
    ) -> pd.DataFrame:
        """
        Compute the utility breakdown for a given case and alternative.

        One and only one of `caseid` or `caseindex` must be given as a keyword
        argument, to indicate which case to compute the utility breakdown for.

        Parameters
        ----------
        altid : int
            The alternative ID on which to compute the utility breakdown.
        caseid : int, optional
            The case ID on which to compute the utility breakdown.
        caseindex : int, optional
            The case index on which to compute the utility breakdown.

        Returns
        -------
        pandas.DataFrame
        """
        if caseid is None and caseindex is None:
            raise ValueError("either caseid or caseindex must be given")
        if caseid is not None and caseindex is not None:
            raise ValueError("only one of caseid or caseindex must be given")
        if caseid is not None:
            caseindex = self.dataset.indexes[self.dataset.dc.CASEID].get_loc(caseid)
        df = {}
        df["co"] = pd.DataFrame(
            [
                (
                    k.param,
                    k.data,
                    float(self.dataset["co"][caseindex].sel(var_co=k.data)),
                    float(self.get_value(k.param)),
                )
                for k in self.utility_co[altid]
            ],
            columns=["parameter_name", "data_expr", "data_value", "parameter_value"],
        )
        df["ca"] = pd.DataFrame(
            [
                (
                    k.param,
                    k.data,
                    float(
                        self.dataset["ca"][caseindex].sel(
                            **{self.dataset.dc.ALTID: altid, "var_ca": k.data}
                        )
                    ),
                    float(self.get_value(k.param)),
                )
                for k in self.utility_ca
            ],
            columns=["parameter_name", "data_expr", "data_value", "parameter_value"],
        )
        result = pd.concat(df, keys=["co", "ca"], names=["utility_type"])
        result["partial_utility"] = result["data_value"] * result["parameter_value"]
        return result.set_index(["utility_type", "parameter_name", "data_expr"])

    def release_memory(self):
        """Release memory-intensive data structures."""
        self._fixed_arrays = None
        self._data_arrays = None
        self._dataset = None
        self.work_arrays = None

    def analyze_predictions_co(
        self,
        q: Any = None,
        n: int = 5,
        *,
        caption: str | bool = True,
        alt_labels: Literal["id", "name"] = "name",
        bins=None,
        wgt: Any = None,
    ) -> pd.io.formats.style.Styler:
        """
        Analyze predictions of the model based on idco attributes.

        This method provides a summary of the model's predictions, broken down
        into categories by some measure in the `idco` data.  The analysis
        includes the mean predicted counts within each category, the standard
        deviation of the predicted counts, the observed values, and a two-tailed
        p-value for the difference between the observed and predicted values.
        Statistically significant differences are highlighted in the output.

        Parameters
        ----------
        q : str or array-like
            The quantiles to use for slicing the data.  If given as a string,
            the string evaluated against the `idca` portion of this model's
            datatree, and then the result is categorized into `n` quantiles.
            If given as an array-like, the array is used to slice the data,
            as the `by` argument to `DataFrame.groupby`, against an `idca`
            formatted dataframe of probabilities.
        n : int, default 5
            The number of quantiles to use when `q` is a string.
        caption : str or bool, default True
            The caption to use for the styled DataFrame.  If True, the caption
            will be "Model Predictions by {q}", and if False no caption will
            be used.
        alt_labels : {'name', 'id'}, default 'name'
            The type of labels to use for the alternative IDs in the output.
        bins : int, sequence of scalars, or IntervalIndex, optional
            If provided, this value overrides `n` and is provided to `pandas.cut`
            to control the binning.

            * int : Defines the number of equal-width bins in the range of `q`. The
              range of `q` is extended by .1% on each side to include the minimum
              and maximum values of `q`.
            * sequence of scalars : Defines the bin edges allowing for non-uniform
              width. No extension of the range of `q` is done.
            * IntervalIndex : Defines the exact bins to be used. Note that
              IntervalIndex for `bins` must be non-overlapping.
        wgt : array-like or str or bool, optional
            If given, this value is used to weight the cases.  This can be done
            whether the model was estimated with weights or not; the estimation
            weights are ignored in this analysis, unless the value of this
            argument is `True`, in which case the estimation weights are used.

        Returns
        -------
        pandas.io.formats.style.Styler
            A styled DataFrame containing the results of the analysis.

        Notes
        -----
        This method is typically used to analyze the model's predictions
        against attributes in the observed data that are not used in the
        model itself.  For example, if the model estimates the probability of
        choosing a particular alternative conditional on cost, time, and
        income, this method can be used to analyze the model's predictions
        against the distribution of observed choices by age or other
        characteristics. Technically, nothing prevents a user from using
        this method to analyze the model's predictions against the same
        attributes used in the model, but the results are likely to provide
        less useful informative.

        This method requires the `scipy` package to be installed, as it uses
        the `scipy.stats.norm.sf` function to calculate the p-values.

        The standard deviation of the predicted counts is calculated via a
        normal approximation to the underlying variable-p binomial-like
        distribution, and may be slightly biased especially for small sample
        sizes.

        """
        try:
            import scipy.stats as stats
        except ImportError as err:
            raise ImportError("scipy is required for this method") from err

        def signif(x):
            return stats.norm.sf(np.absolute(x)) * 2

        def bold_if_signif(value):
            return "font-weight: bold" if value <= 0.05 else ""

        # get probabilities and their variances
        pr = self.probability(return_format="dataframe")
        pr_v = pr * (1 - pr)

        # prepare the slicer, which identifies the groups to analyze
        if isinstance(q, str):
            slicer = self.datatree.idco_subtree().eval(q).single_dim.to_pandas()
        else:
            slicer = q
        if slicer.dtype == bool:
            pass
        elif not isinstance(slicer.dtype, pd.CategoricalDtype):
            name = getattr(slicer, "name", None)
            if bins is not None:
                slicer = pd.cut(slicer, bins)
            else:
                try:
                    slicer = pd.qcut(slicer, n)
                except ValueError as err:
                    if "Bin edges must be unique" in str(err):
                        # maybe there is not enough variation in the data to create
                        # quantiles, if so just convert to categorical
                        if len(slicer.value_counts()) <= n:
                            slicer = pd.Categorical(slicer)
                        else:
                            # otherwise try to drop duplicate bin edges
                            slicer = pd.qcut(slicer, n, duplicates="drop")
                    else:
                        raise
            if name:
                slicer = slicer.rename(name)

        # get the observed values
        obs = self.dataset.ch.to_pandas()

        if caption is True:
            if isinstance(q, str):
                caption = f"Model Predictions by {q}"
            else:
                q_name = getattr(q, "name", None)
                if q_name is None:
                    caption = "Model Predictions"
                else:
                    caption = f"Model Predictions by {q_name}"

        if wgt:
            if isinstance(wgt, str):
                wgt = self.datatree.idco_subtree().eval(wgt).single_dim.to_pandas()
            elif wgt is True:
                wgt = self.dataset.wt.to_pandas()

            w = np.asarray(wgt).reshape(-1, 1)
            # scale obs and probs by the weight
            obs = obs * w
            pr = pr * w
            pr_v = pr_v * w

        result = pd.concat(
            [
                pr.groupby(slicer).sum().stack().rename("mean-predicted"),
                np.sqrt(pr_v.groupby(slicer).sum()).stack().rename("stdev-predicted"),
                obs.groupby(slicer).sum().stack().rename("observed"),
            ],
            axis=1,
        )
        if isinstance(q, str):
            result.index.names = [
                q,
            ] + result.index.names[1:]
        result["signif"] = signif(
            (result["observed"] - result["mean-predicted"]) / result["stdev-predicted"]
        )

        if alt_labels == "name":
            a_map = self.datatree.dc.alts_mapping()
            result.index = result.index.set_levels(
                [a_map.get(i, i) for i in result.index.levels[1]], level=1
            )
            result.index.names = [result.index.names[0], "alt_name"]

        output = (
            result.style.text_gradient(
                cmap="inferno_r",
                axis=0,
                subset=["signif"],
                low=0.5,
                vmin=0.0,
                vmax=0.05,
            )
            .format("{:.3f}")
            .set_table_styles(
                [
                    {"selector": "th", "props": [("border", "1px solid #e5e5e5")]},
                    {"selector": "td", "props": [("border", "1px solid #e5e5e5")]},
                ]
            )
            .applymap(bold_if_signif, subset=["signif"])
        )
        if caption:
            output.set_caption(caption).set_table_styles(
                [
                    {"selector": "th", "props": [("border", "1px solid #e5e5e5")]},
                    {"selector": "td", "props": [("border", "1px solid #e5e5e5")]},
                    {
                        "selector": "caption",
                        "props": [
                            ("text-align", "left"),  # Adjust font size
                            ("font-size", "1.2em"),  # Adjust font size
                            ("font-weight", "bold"),  # Make the caption bold
                            (
                                "padding-bottom",
                                "6px",
                            ),  # Add some space below the caption
                        ],
                    },
                ]
            )
        return output

    def analyze_predictions_co_figure(
        self,
        q: Any = None,
        n: int = 5,
        *,
        caption: str | bool = True,
        signif: float = 0.05,
        width: int = 400,
        alt_labels: Literal["id", "name"] = "name",
        bins=None,
    ) -> altair.vegalite.v5.api.FacetChart:
        """
        Create an Altair figure of the model's predictions based on idco attributes.

        This method provides a summary of the model's predictions, broken down
        into categories by some measure in the `idco` data.  The analysis
        includes the mean predicted counts within each category, the standard
        deviation of the predicted counts, the observed values, and a two-tailed
        p-value for the difference between the observed and predicted values.
        Statistically significant differences are highlighted in the output.

        Parameters
        ----------
        q : str or array-like, optional
            The quantiles to use for slicing the data.  If given as a string,
            the string evaluated against the `idca` portion of this model's
            datatree, and then the result is categorized into `n` quantiles.
            If given as an array-like, the array is used to slice the data,
            as the `by` argument to `DataFrame.groupby`, against an `idca`
            formatted dataframe of probabilities.
        n : int, default 5
            The number of quantiles to use when `q` is a string.
        caption : str or bool, default True
            The caption to use for the figure.  If True, the caption
            will be "Model Predictions by {q}", and if False no caption will
            be used.
        alt_labels : {'id', 'name'}, default 'name'
            The type of labels to use for the alternative IDs in the output.
        signif : float, default 0.05
            The significance level to use for highlighting statistically
            significant differences.
        width : int, default 400
            The width of the figure in pixels.
        bins : int, sequence of scalars, or IntervalIndex, optional
            If provided, this value overrides `n` and is provided to `pandas.cut`
            to control the binning. See `pandas.cut` for more information.

        Returns
        -------
        pandas.io.formats.style.Styler
            A styled DataFrame containing the results of the analysis.

        Notes
        -----
        This method is typically used to analyze the model's predictions
        against attributes in the observed data that are not used in the
        model itself.  For example, if the model estimates the probability of
        choosing a particular alternative conditional on cost, time, and
        income, this method can be used to analyze the model's predictions
        against the distribution of observed choices by age or other
        characteristics. Technically, nothing prevents a user from using
        this method to analyze the model's predictions against the same
        attributes used in the model, but the results are likely to provide
        less useful informative.

        This method requires the `scipy` package to be installed, as it uses
        the `scipy.stats.norm.sf` function to calculate the p-values.

        The standard deviation of the predicted counts is calculated via a
        normal approximation to the underlying variable-p binomial-like
        distribution, and may be slightly biased especially for small sample
        sizes.

        """
        try:
            import scipy.stats as stats
        except ImportError as err:
            raise ImportError("scipy is required for this method") from err

        try:
            import altair as alt
        except ImportError as err:
            raise ImportError("altair is required for this method") from err

        if caption is True:
            if isinstance(q, str):
                caption = f"Model Predictions by {q}"

        df = self.analyze_predictions_co(
            q, n, caption=False, alt_labels=alt_labels, bins=bins
        ).data
        altid_tag = df.index.names[1]
        sort_order = list(df.index.levels[1])
        df = df.reset_index()
        q_ = df.columns[0]
        df[q_] = df[q_].astype(str)

        y = alt.Y(f"{q_}:N", title=q_)
        r = alt.Row(f"{altid_tag}:N", sort=sort_order, title="Alternative")

        threshold = stats.norm.ppf(1 - (signif / 2))
        # Calculate error bar extents
        df["mean-predicted-ciLow"] = np.clip(
            df["mean-predicted"] - df["stdev-predicted"] * threshold, 0, np.inf
        )
        df["mean-predicted-ciHigh"] = (
            df["mean-predicted"] + df["stdev-predicted"] * threshold
        )
        df["Significant"] = np.where(
            df["signif"] <= signif, "Significantly Different", "Consistent"
        )
        df["Constant"] = "Mean"

        tooltip = [
            alt.Tooltip("observed", title="Count Observed"),
            alt.Tooltip("mean-predicted", title="Mean Predicted"),
            alt.Tooltip("stdev-predicted", title="Std Dev Predicted"),
            alt.Tooltip("signif", title="Significance"),
        ]

        # Create base bar chart
        points = (
            alt.Chart(df)
            .mark_point(filled=True, size=40)
            .encode(
                x=alt.X("mean-predicted", title="Number of Observations"),
                y=y,
                tooltip=tooltip,
                color=alt.Color("Constant:N", title="Predicted").scale(range=["black"]),
            )
        )

        # Create error bars
        error_bars = (
            alt.Chart(df)
            .mark_errorbar(ticks=True, color="black")
            .encode(
                x=alt.X("mean-predicted-ciLow:Q", title="Number of Observations"),
                x2="mean-predicted-ciHigh:Q",
                y=y,
                tooltip=tooltip,
            )
        )

        obs_points = (
            alt.Chart(df)
            .mark_point(filled=False, size=60)
            .encode(
                x=alt.X("observed", title="Number of Observations"),
                y=y,
                color=alt.Color(
                    "Significant:N",
                    title="Observed",
                    scale=alt.Scale(
                        domain=["Significantly Different", "Consistent"],
                        range=["red", "#2E8B57"],
                    ),
                ),
                shape=alt.Shape(
                    "Significant:N",
                    title="Observed",
                    scale=alt.Scale(
                        domain=["Significantly Different", "Consistent"],
                        range=["circle", "square"],
                    ),
                ),
                tooltip=tooltip,
            )
        )

        # Combine the layers
        chart = (points + error_bars + obs_points).resolve_scale(
            color="independent", shape="independent"
        )
        return (
            chart.properties(width=width)
            .facet(row=r)
            .resolve_scale(x="independent")
            .properties(title=caption)
            .configure_title(fontSize=16)
        )

    def analyze_elasticity(
        self,
        variable,
        altid: int | None = None,
        q: Any = None,
        n: int = 5,
        *,
        caption: str | bool = True,
        alt_labels: Literal["id", "name"] = "name",
        bins=None,
        wgt: Any = None,
        multiplier: float = 1.01,
        _return_full_probabilities: bool = False,
    ) -> pd.io.formats.style.Styler:
        """
        Analyze elasticity of the model.

        This method provides a summary of the model's aggregate elasticity of
        alternative choices given the data, optionally broken down into segments
        based on chooser attributes.  The computed elasticity is ratio between
        the percentage change in the probability of choosing an alternative
        and the percentage change in the value of the variable of interest. It
        is computed numerically, giving the arc elasticity via finite differences,
        and therefore is agnostic to model structure.

        Parameters
        ----------
        variable : str
            The name of the variable to analyze. This should be a named
            variable in the model's datatree.
        altid : int, optional
            The alternative ID to analyze.  If given, the elasticity is
            calculated for changes in the variable value for this alternative
            only.  If not given, the elasticity is calculated for changes in
            the variable value for all alternatives. This occurs implicitly
            when computing elasticity with respect to `idco` format variables,
            and providing an `altid` for these elasticities will raise an
            error.
        q : str or array-like, optional
            The quantiles to use for slicing the data.  If given as a string,
            the string evaluated against the `idca` portion of this model's
            datatree, and then the result is categorized into `n` quantiles.
            If given as an array-like, the array is used to slice the data,
            as the `by` argument to `DataFrame.groupby`, against an `idca`
            formatted dataframe of probabilities.
        n : int, default 5
            The number of quantiles to use when `q` is a string.
        caption : str or bool, default True
            The caption to use for the styled DataFrame.  If True, the caption
            will be "Model Predictions by {q}", and if False no caption will
            be used.
        alt_labels : {'id', 'name'}, default 'name'
            The type of labels to use for the alternative IDs in the output.
        bins : int, sequence of scalars, or IntervalIndex, optional
            If provided, this value overrides `n` and is provided to `pandas.cut`
            to control the binning.

            * int : Defines the number of equal-width bins in the range of `q`. The
              range of `q` is extended by .1% on each side to include the minimum
              and maximum values of `q`.
            * sequence of scalars : Defines the bin edges allowing for non-uniform
              width. No extension of the range of `q` is done.
            * IntervalIndex : Defines the exact bins to be used. Note that
              IntervalIndex for `bins` must be non-overlapping.
        wgt : array-like or str or bool, optional
            If given, this value is used to weight the cases.  This can be done
            whether the model was estimated with weights or not; the estimation
            weights are ignored in this analysis, unless the value of this
            argument is `True`, in which case the estimation weights are used.

        Returns
        -------
        pandas.io.formats.style.Styler
            A styled DataFrame containing the results of the analysis.

        Notes
        -----
        This method is typically used to analyze the model's predictions
        against attributes in the observed data that are not used in the
        model itself.  For example, if the model estimates the probability of
        choosing a particular alternative conditional on cost, time, and
        income, this method can be used to analyze the model's predictions
        against the distribution of observed choices by age or other
        characteristics. Technically, nothing prevents a user from using
        this method to analyze the model's predictions against the same
        attributes used in the model, but the results are likely to provide
        less useful informative.

        This method requires the `scipy` package to be installed, as it uses
        the `scipy.stats.norm.sf` function to calculate the p-values.

        The standard deviation of the predicted counts is calculated via a
        normal approximation to the underlying variable-p binomial-like
        distribution, and may be slightly biased especially for small sample
        sizes.

        """
        try:
            import scipy.stats as stats
        except ImportError as err:
            raise ImportError("scipy is required for this method") from err

        def signif(x):
            return stats.norm.sf(np.absolute(x)) * 2

        def bold_if_signif(value):
            return "font-weight: bold" if value <= 0.05 else ""

        # get baseline probabilities
        pr = self.probability(return_format="dataframe")

        # get alternate probabilities
        existing_datatree = self.datatree
        temp_var = self.data[variable].copy().astype(float)
        if altid is not None:
            temp_var.loc[{"altid": altid}] *= multiplier
        else:
            temp_var *= multiplier
        self.datatree = self.datatree.replace_datasets(
            {self.datatree.root_node_name: self.data.assign({variable: temp_var})}
        )
        pr_change = self.probability(return_format="dataframe")
        self.datatree = existing_datatree

        if _return_full_probabilities:
            return pr, pr_change

        pr_avg = (pr_change + pr) / 2
        pr_change -= pr

        if caption is True:
            if isinstance(q, str):
                caption = f"Model Elasticity by {q}"
            else:
                q_name = getattr(q, "name", None)
                if q_name is None:
                    caption = "Model Elasticity"
                else:
                    caption = f"Model Elasticity by {q_name}"

        if q is None:
            slicer = None
        else:
            # prepare the slicer, which identifies the groups to analyze
            if isinstance(q, str):
                slicer = self.datatree.idco_subtree().eval(q).single_dim.to_pandas()
            else:
                slicer = q
            if slicer.dtype == bool:
                pass
            elif not isinstance(slicer.dtype, pd.CategoricalDtype):
                name = getattr(slicer, "name", None)
                if bins is not None:
                    slicer = pd.cut(slicer, bins)
                else:
                    try:
                        slicer = pd.qcut(slicer, n)
                    except ValueError as err:
                        if "Bin edges must be unique" in str(err):
                            # maybe there is not enough variation in the data to create
                            # quantiles, if so just convert to categorical
                            if len(slicer.value_counts()) <= n:
                                slicer = pd.Categorical(slicer)
                            else:
                                # otherwise try to drop duplicate bin edges
                                slicer = pd.qcut(slicer, n, duplicates="drop")
                        else:
                            raise
                if name:
                    slicer = slicer.rename(name)

        if wgt:
            if isinstance(wgt, str):
                wgt = self.datatree.idco_subtree().eval(wgt).single_dim.to_pandas()
            elif wgt is True:
                wgt = self.dataset.wt.to_pandas()

            w = np.asarray(wgt).reshape(-1, 1)
            # scale obs and probs by the weight
            pr_change = pr_change * w

        out_name = variable
        a_map = self.datatree.dc.alts_mapping()

        if altid:
            if alt_labels == "name":
                out_name += f"[{a_map.get(altid, altid)}]"
            else:
                out_name += f"[{altid}]"
        if slicer is None:
            result = (pr_change.sum() / pr_avg.sum()).rename(out_name)
        else:
            result = (
                (pr_change.groupby(slicer).sum() / pr_avg.groupby(slicer).sum())
                .stack()
                .rename(out_name)
            )
        result /= multiplier - 1
        if isinstance(q, str):
            result.index.names = [
                q,
            ] + result.index.names[1:]

        if alt_labels == "name":
            if len(result.index.names) > 1:
                result.index = result.index.set_levels(
                    [a_map.get(i, i) for i in result.index.levels[1]], level=1
                )
                result.index.names = [result.index.names[0], "alt_name"]
            else:
                result.index = [a_map.get(x, x) for x in result.index]
                result.index.name = "alt_name"

        output = (
            pd.DataFrame(result)
            .style.format("{:.3f}")
            .set_table_styles(
                [
                    {"selector": "th", "props": [("border", "1px solid #e5e5e5")]},
                    {"selector": "td", "props": [("border", "1px solid #e5e5e5")]},
                ]
            )
        )
        if caption:
            output.set_caption(caption).set_table_styles(
                [
                    {"selector": "th", "props": [("border", "1px solid #e5e5e5")]},
                    {"selector": "td", "props": [("border", "1px solid #e5e5e5")]},
                    {
                        "selector": "caption",
                        "props": [
                            ("text-align", "left"),  # Adjust font size
                            ("font-size", "1.2em"),  # Adjust font size
                            ("font-weight", "bold"),  # Make the caption bold
                            (
                                "padding-bottom",
                                "6px",
                            ),  # Add some space below the caption
                        ],
                    },
                ]
            )
        return output


@njit(cache=True)
def _arr_inflate(arr, locks):
    s = locks.size
    z = np.zeros((s, s), dtype=arr.dtype)
    i_ = 0
    for i in range(s):
        if not locks[i]:
            j_ = 0
            for j in range(s):
                if not locks[j]:
                    z[i, j] = arr[i_, j_]
                    j_ += 1
            i_ += 1
    return z


def _safe_sqrt(x):
    result = np.sqrt(np.clip(x, 0, np.inf))
    result[x < 0] = np.nan
    return result

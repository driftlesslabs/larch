from __future__ import annotations

import logging
import pathlib
import warnings
from collections import namedtuple

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
from ..util import dictx
from ..util.simple_attribute import SimpleAttribute
from .basemodel import BaseModel as _BaseModel
from .numba_stream import ModelStreamer

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
                    dutility_elem[
                        altindex, model_utility_co_param[i]
                    ] += model_utility_co_param_scale[i]
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
                    dutility[up, :] += cond_prob * dutility[dn, :]

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

    def mangle(self, data=True, structure=True):
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

    def reflow_data_arrays(self):
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
                self._data_arrays = self.dataset.dc.to_arrays(
                    self.graph,
                    float_dtype=self.float_dtype,
                )
                if self.work_arrays is not None:
                    self._rebuild_work_arrays()

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
            nameset, value=1, nullvalue=1, initvalue=1, minimum=0.001, maximum=1
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
                    self.work_arrays.loglike[caseslice] += penalty
                    self.work_arrays.d_loglike[caseslice] += np.expand_dims(dpenalty, 0)
                    self.work_arrays.bhhh[caseslice] = np.einsum(
                        "ij,ik->ijk",
                        self.work_arrays.d_loglike[caseslice],
                        self.work_arrays.d_loglike[caseslice],
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
        self, x=None, *, start_case=None, stop_case=None, step_case=None, **kwargs
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

    def robust_covariance(self):
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
            "robust_std_err", np.sqrt(np.diagonal(robust_covariance_matrix))
        )
        return self.parameters["robust_covariance_matrix"]

    def _wrap_as_dataframe(
        self,
        arr,
        return_dataframe,
        start_case=None,
        stop_case=None,
        step_case=None,
    ):
        if return_dataframe:
            idx = self.datatree.caseids()
            if idx is not None:
                idx = idx[start_case:stop_case:step_case]
            if return_dataframe == "names":
                return pd.DataFrame(
                    data=arr,
                    columns=self.graph.standard_sort_names[: arr.shape[1]],
                    index=idx,
                )
            if return_dataframe == "dataarray":
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
                columns=self.graph.standard_sort[: arr.shape[1]],
                index=idx,
            )
            if return_dataframe == "idce":
                raise NotImplementedError
            elif return_dataframe == "idca":
                return result.stack()
            else:
                return result
        return arr

    def probability(
        self,
        x=None,
        *,
        start_case=None,
        stop_case=None,
        step_case=None,
        return_dataframe=False,
        include_nests=False,
    ):
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
            return_dataframe,
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
        return_dataframe=None,
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
            return_dataframe,
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
        leave_out=-1,
        keep_only=-1,
        subsample=-1,
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
        leave_out=-1,
        keep_only=-1,
        subsample=-1,
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
        leave_out=-1,
        keep_only=-1,
        subsample=-1,
    ):
        if x is None:
            x = self.pvals.copy()
        from ..util.math import approx_fprime

        return approx_fprime(
            x,
            lambda y: self.d_loglike(
                y,
                start_case=start_case,
                stop_case=stop_case,
                step_case=step_case,
                leave_out=leave_out,
                keep_only=keep_only,
                subsample=subsample,
            ),
        )

    def neg_loglike(
        self,
        x=None,
        start_case=None,
        stop_case=None,
        step_case=None,
        leave_out=-1,
        keep_only=-1,
        subsample=-1,
    ):
        result = self.loglike(
            x,
            start_case=start_case,
            stop_case=stop_case,
            step_case=step_case,
            leave_out=leave_out,
            keep_only=keep_only,
            subsample=subsample,
        )
        return -result

    def logloss(
        self,
        x=None,
        start_case=None,
        stop_case=None,
        step_case=None,
        leave_out=-1,
        keep_only=-1,
        subsample=-1,
    ):
        result = self.loglike(
            x,
            start_case=start_case,
            stop_case=stop_case,
            step_case=step_case,
            leave_out=leave_out,
            keep_only=keep_only,
            subsample=subsample,
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
        leave_out=-1,
        keep_only=-1,
        subsample=-1,
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
                    leave_out=leave_out,
                    keep_only=keep_only,
                    subsample=subsample,
                )
                current_dll = result.dll
                current_bhhh = result.bhhh
                bhhh_inv = self._free_slots_inverse_matrix(current_bhhh)
                direction = np.dot(current_dll, bhhh_inv)
                j_pvals += direction * steplen
                logger.debug(f"jump to {j_pvals}")
                self.set_values(j_pvals)

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
            "dataflows",
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
    def n_cases(self):
        """Int : The number of cases in the attached data."""
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
    def dataset(self):
        """larch.Dataset : Data arrays as loaded for model computation."""
        super().unmangle()
        if self._dataset is None:
            self.reflow_data_arrays()
        try:
            return self._dataset
        except AttributeError:
            return None

    @dataset.setter
    def dataset(self, dataset):
        if dataset is self._dataset:
            return
        from xarray import Dataset as _Dataset

        if isinstance(dataset, Dataset):
            self._dataset = dataset
            self._data_arrays = None
        elif isinstance(dataset, _Dataset):
            self._dataset = Dataset(dataset)
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
    ):
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
        dictx
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

    def calculate_parameter_covariance(self, pvals=None, *, robust=False):
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
                ihess_ = np.linalg.inv(hess_)
                ihess = _arr_inflate(ihess_, locks)
            else:
                ihess = np.linalg.inv(hess)
            se = np.sqrt(ihess.diagonal())
            self.pstderr = se
        hess = np.asarray(hess).copy()
        hess[locks, :] = 0
        hess[:, locks] = 0
        ihess = np.asarray(ihess).copy()
        ihess[locks, :] = 0
        ihess[:, locks] = 0
        self.add_parameter_array("hess", hess)
        self.add_parameter_array("ihess", ihess)

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
            self.pstderr = np.sqrt(s.diagonal())

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

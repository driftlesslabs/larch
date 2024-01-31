from __future__ import annotations

import numba as nb
import numpy as np
import pandas as pd
from sharrow import Flow


@nb.njit(nogil=True)
def cascade_sum_1row(arr, dn_slots, up_slots, n_nodes, max_val=None):
    out = np.zeros(n_nodes, dtype=arr.dtype)
    out[: arr.size] = arr[:]
    for j in range(dn_slots.size):
        if max_val is None:
            out[up_slots[j]] += out[dn_slots[j]]
        else:
            k = out[up_slots[j]] + out[dn_slots[j]]
            out[up_slots[j]] = min(k, max_val)
    return out


def _raw_streamer(source, name, dtype):
    raw_data = source.root_dataset[name].to_numpy()
    if raw_data.dtype == dtype:
        func = lambda c: raw_data[c]
    else:
        func = lambda c: raw_data[c].astype(dtype)
    return nb.njit(func)


class OneHotStreamer:
    def __init__(self, key, coords):
        self.key = key
        self.coords = coords

    def compile(self, source, dtype):
        raw_codes = source._getitem(self.key).to_numpy()
        n_alts = source.n_alts
        coords = np.asarray(self.coords)

        def pseudo_flow(c):
            out = np.zeros(n_alts, dtype=dtype)
            this_code = raw_codes[c]
            for i in range(coords.size):
                if this_code == coords[i]:
                    out[i] = 1
                    break
            return out

        return nb.njit(pseudo_flow)


def get_data_streamers(model, default_dtype=np.float64):
    dtypes = {"av": np.int8, "avail_ca": np.int8}
    streamers = {}
    dataflows = model.dataflows
    for k, v in dataflows.items():
        if k in {"co", "wt"}:
            source = model.datatree.idco_subtree()
        else:
            source = model.datatree
        if isinstance(v, Flow):
            streamers[k] = source.reuse_streamer(v, dtype=dtypes.get(k, default_dtype))
            # streamers[k] = v.init_streamer(
            #     source=source,
            #     dtype=dtypes.get(k, default_dtype),
            #     closure=True,
            # )
        elif isinstance(v, str):
            streamers[k] = _raw_streamer(source, v, dtypes.get(k, default_dtype))
        elif isinstance(v, OneHotStreamer):
            streamers[k] = v.compile(source, dtypes.get(k, default_dtype))
        else:
            raise NotImplementedError
    return streamers


def init_streamer(
    model,
):
    from .numbamodel import (  # inside function to prevent circular import
        _numba_utility_to_loglike,
        quantity_from_data_ca,
        utility_from_data_ca,
        utility_from_data_co,
    )

    streamers = get_data_streamers(model, model.float_dtype)
    float_dtype = model.float_dtype
    n_alts = model.datatree.n_alts

    if "ca" in streamers:
        stream_ca = streamers["ca"].compiled
    else:
        stream_ca = nb.njit(lambda c: np.zeros((n_alts, 0), dtype=float_dtype))

    if "co" in streamers:
        stream_co = streamers["co"].compiled
    else:
        stream_co = nb.njit(lambda c: np.zeros((0,), dtype=float_dtype))

    if "avail_ca" in streamers:
        _stream_av_1 = streamers["avail_ca"]
        _stream_av = streamers["avail_ca"].compiled
        stream_av = nb.njit(lambda c: np.reshape(_stream_av(c), -1))
    elif model.availability_any:
        stream_av = nb.njit(lambda c: np.ones((n_alts), dtype=np.int8))
    else:
        raise NotImplementedError

    if "choice_ca" in streamers:
        _stream_ch = streamers["choice_ca"].compiled
        stream_ch = nb.njit(lambda c: np.reshape(_stream_ch(c), -1))
    elif "choice_co_code" in streamers:
        stream_ch = streamers["choice_co_code"]
    else:
        raise NotImplementedError

    if "weight_co" in streamers:
        stream_wt = streamers["weight_co"].compiled
    else:
        stream_wt = nb.njit(lambda c: np.ones((1,), dtype=float_dtype))

    @nb.njit
    def streamer(
        c,
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
        # array_ch,  # [17] float input shape=[nodes]
        # array_av,  # [18] int8 input shape=[nodes]
        # array_wt,  # [19] float input shape=[]
        # array_co,  # [20] float input shape=[n_co_vars]
        # array_ca,  # [21] float input shape=[n_alts, n_ca_vars]
        # array_ce_data,  # [22] float input shape=[n_casealts, n_ca_vars]
        # array_ce_indices,  # [23] int input shape=[n_casealts]
        # array_ce_ptr,  # [24] int input shape=[2]
        return_flags,
        # utility,  # [23] float output shape=[nodes]
        # logprob,  # [24] float output shape=[nodes]
        # probability,  # [25] float output shape=[nodes]
        # bhhh,  # [26] float output shape=[n_params, n_params]
        # d_loglike,  # [27] float output shape=[n_params]
        # loglike,  # [28] float output shape=[]
    ):
        array_ch_ = stream_ch(c)  # float input shape=[nodes]
        array_av_ = stream_av(c)  # int8 input shape=[nodes]
        array_wt = stream_wt(c)  # float input shape=[]
        array_co = stream_co(c)  # float input shape=[n_co_vars]
        array_ca = stream_ca(c)  # float input shape=[n_alts, n_ca_vars]

        n_alts = array_ca.shape[0]
        n_nodes = mu_slots.shape[0] + n_alts

        array_av = cascade_sum_1row(
            array_av_, edgeslots[:, 1], edgeslots[:, 0], n_nodes, 127
        )
        array_ch = cascade_sum_1row(
            array_ch_, edgeslots[:, 1], edgeslots[:, 0], n_nodes
        )

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

        utility = np.zeros(n_nodes, dtype=float_dtype)
        dutility = np.zeros((utility.size, parameter_arr.size), dtype=utility.dtype)
        n_params = parameter_arr.shape[0]
        logprob = np.zeros(n_nodes, dtype=float_dtype)
        probability = np.zeros(n_nodes, dtype=float_dtype)
        bhhh = np.zeros((n_params, n_params), dtype=float_dtype)
        d_loglike = np.zeros(n_params, dtype=float_dtype)
        loglike = np.zeros(1, dtype=float_dtype)

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
            return utility, dutility, logprob, probability, bhhh, d_loglike, loglike

        # TODO not zeros?
        array_ce_data = np.zeros((0, array_ca.shape[1]), dtype=array_ca.dtype)
        array_ce_indices = np.zeros((0,), dtype=np.int32)
        array_ce_ptr = np.zeros((2,), dtype=np.int32)

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
            return utility, dutility, logprob, probability, bhhh, d_loglike, loglike

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

        return utility, dutility, logprob, probability, bhhh, d_loglike, loglike

    return streamer


def init_choice_avail_summary_streamer(model):
    streamers = get_data_streamers(model, model.float_dtype)
    float_dtype = model.float_dtype
    n_alts = model.datatree.n_alts

    if "ca" in streamers:
        stream_ca = streamers["ca"]  # noqa: F841
    else:
        stream_ca = nb.njit(lambda c: np.zeros((1, n_alts, 0), dtype=float_dtype))  # noqa: F841

    if "co" in streamers:
        stream_co = streamers["co"]  # noqa: F841
    else:
        stream_co = nb.njit(lambda c: np.zeros((1, 0), dtype=float_dtype))  # noqa: F841

    if "avail_ca" in streamers:
        _stream_av_1 = streamers["avail_ca"]
        _stream_av = streamers["avail_ca"].compiled
        stream_av = nb.njit(lambda c: np.reshape(_stream_av(c), -1))
    elif model.availability_any:
        stream_av = nb.njit(lambda c: np.ones((n_alts), dtype=np.int8))
    else:
        raise NotImplementedError

    if "choice_ca" in streamers:
        _stream_ch = streamers["choice_ca"].compiled
        stream_ch = nb.njit(lambda c: np.reshape(_stream_ch(c), -1))
    elif "choice_co_code" in streamers:
        stream_ch = streamers["choice_co_code"]
    else:
        raise NotImplementedError

    if "weight_co" in streamers:
        stream_wt = streamers["weight_co"]
    else:
        stream_wt = nb.njit(lambda c: np.ones((1,), dtype=float_dtype))

    @nb.njit(parallel=False, nogil=True)
    def array_streamer(
        n_cases,
        edgeslots,  # [11] int input shape=[edges, 4]
        mu_slots,  # [12] int input shape=[nests]
        n_alts,
    ):
        n_nodes = mu_slots.shape[0] + n_alts
        total_ch = np.zeros(n_nodes, dtype=np.float32)
        total_av = np.zeros(n_nodes, dtype=np.float32)
        total_wt = np.zeros((1,), dtype=np.float32)
        eslots1 = edgeslots[:, 1]
        eslots0 = edgeslots[:, 0]
        for c in range(n_cases):
            total_av += cascade_sum_1row(stream_av(c), eslots1, eslots0, n_nodes, 127)
            n_ch = cascade_sum_1row(stream_ch(c), eslots1, eslots0, n_nodes)
            total_ch += n_ch
            total_wt += stream_wt(c) * n_ch[-1]  # float input shape=[]
        return total_ch, total_av, total_wt

    return array_streamer


@nb.njit(nogil=True, cache=True)
def _ll_streaming(
    pvals,
    c,
    _streamfunc,
    _fixed_arrays,
    _pholdfast,
    return_grad=True,
    return_bhhh=False,
):
    utility, dutility, logprob, probability, bhhh, d_loglike, loglike = _streamfunc(
        c,
        *_fixed_arrays,
        _pholdfast,
        pvals,
        np.asarray(
            [
                False,  # only_utility
                False,  # return_probability
                return_grad,
                return_bhhh,
            ],
            dtype=np.int8,
        ),
    )
    return loglike[0], d_loglike, bhhh


@nb.njit(parallel=True, nogil=True, cache=True)
def _loglike_streaming(
    pvals, start_case, stop_case, _streamfunc, _fixed_arrays, _pholdfast
):
    result = 0
    for c in nb.prange(start_case, stop_case):
        ll, _, _ = _ll_streaming(
            pvals,
            c,
            _streamfunc,
            _fixed_arrays,
            _pholdfast,
            return_grad=False,
            return_bhhh=False,
        )
        result += ll
    return result


@nb.njit(parallel=True, nogil=True, cache=True)
def _d_loglike_streaming(
    pvals, start_case, stop_case, _streamfunc, _fixed_arrays, _pholdfast
):
    result = 0
    d_result = np.zeros_like(pvals)
    for c in nb.prange(start_case, stop_case):
        ll, dll, _ = _ll_streaming(
            pvals,
            c,
            _streamfunc,
            _fixed_arrays,
            _pholdfast,
            return_grad=True,
            return_bhhh=False,
        )
        result += ll
        d_result += dll
    return result, d_result


class ModelStreamer:
    def __init__(self, model=None):
        self._model = model
        # if model is None:
        #     return
        # if not hasattr(self._model, "_streamfunc"):
        #     self._model._streamfunc = init_streamer(model)
        # _fixed_arrays = tuple(self._model._fixed_arrays)
        # _pholdfast = self._model.pholdfast
        #
        # self._choice_avail_summary_streamer = init_choice_avail_summary_streamer(
        #     self._model
        # )

        # def _loglike_func_py(pvals, c, _streamfunc, return_grad=True, return_bhhh=False):
        #     utility, dutility, logprob, probability, bhhh, d_loglike, loglike = _streamfunc(
        #         c, *_fixed_arrays, _pholdfast, pvals, np.asarray(
        #             [
        #                 False,  # only_utility
        #                 False,  # return_probability
        #                 return_grad,
        #                 return_bhhh,
        #             ],
        #             dtype=np.int8,
        #         ))
        #     return loglike[0], d_loglike, bhhh
        # self._loglike_case = _loglike_func_nb = nb.njit(_loglike_func_py)

        # def _loglike_sum(pvals, start_case, stop_case, _streamfunc):
        #     result = 0
        #     for c in nb.prange(start_case, stop_case):
        #         result += _loglike_func_nb(pvals, c, _streamfunc)
        #     return result
        # print("init _loglike_sum")
        # self._loglike_sum = nb.njit(_loglike_sum, parallel=True)

    def reset_streamer(self):
        self._model._streamfunc = init_streamer(self._model)

    def __set_name__(self, owner, name):
        self.name = name
        self.private_name = "_" + name

    def __get__(self, obj, objtype=None):
        """
        Get the value of an attribute.

        Parameters
        ----------
        obj : Any
            Instance of parent class that has `self` as a member.
        objtype : type
            Class of `obj`.
        """
        if obj is None:
            return self
        s = getattr(obj, self.private_name, None)
        if s is None:
            obj.unmangle(structure_only=True)
            s = type(self)(obj)
            setattr(obj, self.private_name, s)
        return s

    def __set__(self, instance, value):
        """
        Set the value of an attribute.

        Parameters
        ----------
        instance : Any
            Instance of parent class that has `self` as a member.
        value : Any
            Value being set.
        """
        if value is not None:
            raise NotImplementedError
        setattr(instance, self.private_name, value)
        # existing_value = self.__get__(instance)
        # if value is not None:
        #     value = str(value)
        # if existing_value != value:
        #     setattr(instance, self.private_name, value)
        #     try:
        #         instance.mangle()
        #     except AttributeError:
        #         pass

    def __delete__(self, instance):
        self.__set__(instance, None)

    def loglike(self, x=None, *, start_case=0, stop_case=-1, step_case=1, **kwargs):
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
        if start_case is None:
            start_case = 0
        if stop_case is None:
            stop_case = -1
        if step_case is None:
            step_case = 1
        if step_case != 1:
            raise NotImplementedError
        if stop_case < 0:
            stop_case = self._model.n_cases + 1 + stop_case
        if x is not None:
            self._model.pvals = x
        _fixed_arrays = tuple(self._model._fixed_arrays)
        _pholdfast = self._model.pholdfast
        return _loglike_streaming(
            self._model.pvals,
            start_case,
            stop_case,
            self._model._streamfunc,
            _fixed_arrays,
            _pholdfast,
        )

    def d_loglike(self, x=None, *, start_case=0, stop_case=-1, step_case=1, **kwargs):
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
        if start_case is None:
            start_case = 0
        if stop_case is None:
            stop_case = -1
        if step_case is None:
            step_case = 1
        if step_case != 1:
            raise NotImplementedError
        if stop_case < 0:
            stop_case = self._model.n_cases + 1 + stop_case
        if x is not None:
            self._model.pvals = x
        _fixed_arrays = tuple(self._model._fixed_arrays)
        _pholdfast = self._model.pholdfast
        return _d_loglike_streaming(
            self._model.pvals,
            start_case,
            stop_case,
            self._model._streamfunc,
            _fixed_arrays,
            _pholdfast,
        )[1]

    def choice_avail_summary(self):
        total_ch, total_av, total_wt = self._choice_avail_summary_streamer(
            self._model.n_cases,
            self._model._fixed_arrays.edge_slots,
            self._model._fixed_arrays.mu_slot,
            self._model.datatree.n_alts,
        )
        od = {}
        graph = self._model.graph
        idx = pd.Index(graph.standard_sort, name="altid")
        od["name"] = pd.Series(graph.standard_sort_names, index=idx)
        od["chosen"] = pd.Series(total_ch, index=idx)
        od["available"] = pd.Series(total_av, index=idx)
        result = pd.DataFrame(od, index=idx)
        from ..dataset.choice_avail_reporting import clean_summary

        return clean_summary(result, root_id=graph.root_id)

    def total_weight(self):
        if getattr(self, "_total_weight", None) is None:
            total_ch, total_av, total_wt = self._choice_avail_summary_streamer(
                self._model.n_cases,
                self._model._fixed_arrays.edge_slots,
                self._model._fixed_arrays.mu_slot,
                self._model.datatree.n_alts,
            )
            self._total_weight = float(total_wt)
        return self._total_weight

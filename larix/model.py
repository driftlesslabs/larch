import jax
import jax.numpy as jnp
import larch.numba as lx
import numpy as np
import pandas as pd

from larch.numba import Dataset, DataArray

from .compiled import compiledmethod, jitmethod, reset_compiled_methods
from .folding import fold_dataset
from .optimize import OptimizeMixin

def _get_jnp_array(dataset, name):
    if name not in dataset:
        return None
    return jnp.asarray(dataset[name])

class Model(lx.Model, OptimizeMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mixtures = []
        self._n_draws = 100
        self._draws = None
        self._groupid = None
        self._parameter_bucket = None

    @property
    def groupid(self):
        return self._groupid

    @groupid.setter
    def groupid(self, g):
        if isinstance(g, str) and g == self._groupid:
            return
        else:
            self._groupid = g
            self.mangle()

    @property
    def n_draws(self):
        return self._n_draws

    @n_draws.setter
    def n_draws(self, n):
        if n == self._n_draws:
            return
        else:
            self._n_draws = n
            self.mangle()

    @property
    def pvals(self):
        if self._parameter_bucket is None:
            return super().pvals
        else:
            return self._parameter_bucket.pvals

    @pvals.setter
    def pvals(self, x):
        if self._parameter_bucket is None:
            self.set_values(x)
        else:
            self._parameter_bucket.pvals = x

    @property
    def pstderr(self):
        if self._parameter_bucket is None:
            if "std_err" in self._frame:
                return self._frame['std_err'].to_numpy()
            else:
                return np.full_like(self.pvals, np.nan)
        else:
            return self._parameter_bucket.pstderr

    @pstderr.setter
    def pstderr(self, x):
        if self._parameter_bucket is None:
            self._frame['std_err'] = x
        else:
            self._parameter_bucket.pstderr = x

    def mangle(self, *args, **kwargs):
        super().mangle(*args, **kwargs)
        self._draws = None
        reset_compiled_methods(self)

    def unmangle(self, force=False):
        super().unmangle(force=force)
        for mix in self._mixtures:
            mix.mean = self._frame.index.get_loc(mix.mean_)
            mix.std = self._frame.index.get_loc(mix.std_)
        if self.groupid is not None:
            self.dataset = fold_dataset(self.dataset, self.groupid)

    def reflow_data_arrays(self):
        """
        Reload the internal data_arrays so they are consistent with the datatree.
        """
        if self.graph is None:
            self._data_arrays = None
            return

        datatree = self.datatree
        if datatree is not None:
            from larch.numba.data_arrays import prepare_data
            self.dataset, self.dataflows = prepare_data(
                datasource=datatree,
                request=self,
                float_dtype=self.float_dtype,
                cache_dir=datatree.cache_dir,
                flows=getattr(self, 'dataflows', None),
            )
            self._data_arrays = self.dataset.to_arrays(
                self.graph,
                float_dtype=self.float_dtype,
            )
            if self.work_arrays is not None:
                self._rebuild_work_arrays()

        elif self.dataframes is not None: # work from old DataFrames
            raise NotImplementedError("use Dataset not Dtaframes")

    @property
    def data_as_loaded(self):
        return self._dataset

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
            if self.groupid is not None:
                dataset = fold_dataset(dataset, self.groupid)
            self._dataset = dataset
            self._data_arrays = None
            self._rebuild_fixed_arrays()
        elif isinstance(dataset, _Dataset):
            if self.groupid is not None:
                dataset = fold_dataset(dataset, self.groupid)
            self._dataset = Dataset(dataset)
            self._data_arrays = None
            self._rebuild_fixed_arrays()
        else:
            raise TypeError(f"dataset must be Dataset not {type(dataset)}")

    def set_value(self, name, value=None, **kwargs):
        """
        Set the value for a single model parameter.

        This function will set the current value of a parameter.
        Unless explicitly instructed with an alternate value,
        the new value will also be saved as the "initial" value
        of the parameter.

        Parameters
        ----------
        name : str
            The name of the parameter to set to a fixed value.
        value : float
            The numerical value to set for the parameter.
        initvalue : float, optional
            If given, this value is used to indicate the initial value
            for the parameter, which may be different from the
            current value.
        nullvalue : float, optional
            If given, this will overwrite any existing null value for
            the parameter.  If not given, the null value for the
            parameter is not changed.

        """
        dtypes = self._frame.dtypes.copy()
        name = str(name)
        if name not in self._frame.index:
            self.unmangle()
        if value is not None:
            kwargs['value'] = value
        for k, v in kwargs.items():
            if k in self._frame.columns:
                if dtypes[k] == 'float64':
                    v = np.float64(v)
                elif dtypes[k] == 'float32':
                    v = np.float32(v)
                elif dtypes[k] == 'int8':
                    v = np.int8(v)
                self._frame.loc[name, k] = v

            # update init values when they are implied
            if k == 'value':
                if 'initvalue' not in kwargs and pd.isnull(self._frame.loc[name, 'initvalue']):
                    self._frame.loc[name, 'initvalue'] = np.float32(v)

        # update null values when they are implied
        if 'nullvalue' not in kwargs and pd.isnull(self._frame.loc[name, 'nullvalue']):
            self._frame.loc[name, 'nullvalue'] = 0

        self._frame = self._frame.astype(dtypes)

        self._check_if_frame_values_changed()

    def mix_parameter(self, *args, distribution='normal', cap=50):
        if distribution == 'normal':
            from .mixtures import Normal

            if args[1] not in self.pf.index:
                self.set_value(
                    args[1],
                    0.001,
                    minimum=-cap,
                    maximum=cap,
                    holdfast=0,
                    note='',
                )

            m_ = self.pf.index.get_loc(args[0])
            s_ = self.pf.index.get_loc(args[1])
            self._mixtures.append(Normal(args[0], args[1], m_, s_))
            self.mangle()
        else:
            raise ValueError(f"unknown distribution {distribution}")

    def _make_random_draws(self, n_draws, seed=0):
        random_key = jax.random.PRNGKey(seed)
        self._draws = jax.random.uniform(
            random_key,
            [n_draws, len(self._mixtures)],
        )

        def body(i, carry):
            draws, rnd_key = carry
            rnd_key, subkey = jax.random.split(rnd_key)
            draws = draws.at[:, i].add(jax.random.permutation(subkey, draws.shape[0]))
            return draws, rnd_key

        self._draws, random_key = jax.lax.fori_loop(0, self._draws.shape[1], body, (self._draws, random_key))
        self._draws = self._draws * (1 / n_draws)

    def apply_random_draws(self, parameters):
        parameters = jnp.broadcast_to(
            parameters,
            [self._draws.shape[0], parameters.shape[0]]
        )
        for mix_n, mix in enumerate(self._mixtures):
            u = self._draws[:, mix_n]
            parameters = mix.roll(u, parameters)
        return parameters

    @jitmethod
    def _jax_utility(self, params, ca=None, co=None, av=None):
        n_alts = self.dataset.n_alts
        n_nodes = len(self.graph)
        x = jnp.zeros([self.dataset.n_alts, self.dataset.dims.get("var_co", 0) + 1])
        x = x.at[self._fixed_arrays.uco_alt_slot, self._fixed_arrays.uco_data_slot].add(
            params[self._fixed_arrays.uco_param_slot]
        )
        u = jnp.zeros([n_nodes])
        if ca is not None:
            u = u.at[:n_alts].add(jnp.dot(
                ca[..., self._fixed_arrays.uca_data_slot],
                params[self._fixed_arrays.uca_param_slot],
            ))
        if co is not None:
            u = u.at[:n_alts].add(
                jnp.dot(co, x[:, :-1].T)
            )
        u = u.at[:n_alts].add(
            x[:, -1].T
        )
        if av is not None:
            u = u.at[:n_alts].set(
                jnp.where(av[:n_alts], u[:n_alts], -jnp.inf)
            )
        return u

    @jitmethod
    def jax_utility(self, params):
        n_alts = self.dataset.n_alts
        n_nodes = len(self.graph)

        ca = _get_jnp_array(self.dataset, "ca")
        co = _get_jnp_array(self.dataset, "co")
        av = _get_jnp_array(self.dataset, "av")

        if av is not None:
            n_cases = av.shape[:-1]
        elif co is not None:
            n_cases = co.shape[:-1]
        elif ca is not None:
            n_cases = ca.shape[:-2]

        x = jnp.zeros([self.dataset.n_alts, self.dataset.dims.get("var_co", 0) + 1])
        x = x.at[self._fixed_arrays.uco_alt_slot, self._fixed_arrays.uco_data_slot].add(
            params[self._fixed_arrays.uco_param_slot]
        )

        u = jnp.zeros([*n_cases, n_nodes])
        if ca is not None:
            u = u.at[..., :n_alts].add(jnp.dot(
                ca[..., self._fixed_arrays.uca_data_slot],
                params[self._fixed_arrays.uca_param_slot],
            ))
        if co is not None:
            u = u.at[..., :n_alts].add(
                jnp.dot(co, x[:, :-1].T)
            )
        u = u.at[..., :n_alts].add(
            x[:, -1].T
        )
        if av is not None:
            u = u.at[..., :n_alts].set(
                jnp.where(av[..., :n_alts], u[..., :n_alts], -jnp.inf)
            )
        return u

    def __utility_for_nest(self, slot):
        nest_code = self.graph.standard_sort[slot]
        child_slots = self.graph.successor_slots(nest_code)
        mu_name = self.graph.nodes[nest_code].get('parameter')
        if mu_name is None:
            mu_slot = -1
        else:
            mu_slot = self.pf.index.get_loc(mu_name)

        #@jit
        def u_nest(params, utility_array, array_av):
            mu = params[mu_slot] if mu_slot >= 0 else 1.0
            carry = jnp.zeros(utility_array.shape[:-1])
            #num_av = jnp.sum(utility_array[..., child_slots] > -1e30, axis=-1)
            num_av = array_av[...,slot]

            def body(carry, child_slot):
                carry = jnp.add(
                    carry,
                    jnp.where(
                        num_av > 1,
                        jnp.exp(jnp.clip(utility_array[..., child_slot], -1e37) / mu),
                        jnp.exp(jnp.clip(utility_array[..., child_slot], -1e37)),
                    )
                )
                return carry, None

            carry, _ = jax.lax.scan(body, carry, child_slots)
            utility_array = utility_array.at[..., slot].add(
                jnp.where(
                    num_av > 1,
                    jnp.clip(jnp.log(carry), -1e38) * mu,
                    jnp.clip(jnp.log(carry), -1e39),
                )
            )
            return utility_array

        return u_nest

    @compiledmethod
    def utility_for_nests(self):
        n_alts = self.graph.n_elementals()
        n_nodes = len(self.graph)

        def u_nesting_none(out, beta, array_av):
            # TODO Maybe filter on av
            return out.at[...,-1].set(
                jnp.log(jnp.exp(out[...,:-1]).sum(-1))
            )

        if n_nodes - n_alts <= 1:
            return u_nesting_none

        #@jit
        def u_nesting_few(out, beta, array_av):
            for slot in range(n_alts, n_nodes):
                out = self.__utility_for_nest(slot)(beta, out, array_av)
            return out

        if n_nodes - n_alts < 3:
            return u_nesting_few

        # many nests, use more efficient loop
        n_params = len(self.pf)
        mu_names = [self.graph.nodes[i].get('parameter') for i in self.graph.standard_sort]
        mu_slots = jnp.asarray(
            [(self.pf.index.get_loc(mu_name) if mu_name is not None else n_params) for mu_name in mu_names]
        )
        def add_u_to_parent(u, params, child_slot, parent_slot, avail):
            mu = params[mu_slots[parent_slot]]
            u = u.at[...,parent_slot].add(
                jnp.where(
                    avail[...,parent_slot] > 1,
                    jnp.exp(jnp.clip(u[..., child_slot], -1e37) / mu),
                    jnp.exp(jnp.clip(u[..., child_slot], -1e37)     ),
                )
            )
            return u
        def log_self(u, params, self_slot, avail):
            mu = params[mu_slots[self_slot]]
            u = u.at[..., self_slot].set(
                jnp.where(
                    avail[..., self_slot] > 1,
                    jnp.clip(jnp.log(u[..., self_slot]), -1e38) * mu,
                    jnp.clip(jnp.log(u[..., self_slot]), -1e38)
                )
            )
            return u
        slotarray = np.stack(self.graph.edge_slot_arrays()).T

        @jax.jit
        def u_rollup(utility_array, parameter_vector, avail_ca):
            n_params = parameter_vector.size
            params = jnp.ones(n_params + 1, dtype=parameter_vector.dtype)
            params = params.at[:n_params].set(parameter_vector)

            def body(carry, xs):
                u, params = carry
                up_slot, dn_slot, firstvisit, allocslot = xs
                # if firstvisit >= 0 and dn_slot>=n_alts:
                #     u = log_self(u, params, dn_slot)
                u = jax.lax.cond(
                    (firstvisit >= 0) & (dn_slot>=n_alts),
                    lambda u: log_self(u, params, dn_slot, avail_ca),
                    lambda u: u,
                    operand=u
                )
                u = add_u_to_parent(u, params, dn_slot, up_slot, avail_ca)
                return (u, params), None
            (utility_array, _ignore_1), _ignore_2 = jax.lax.scan(body, (utility_array, params), slotarray)
            # log utility at root
            utility_array = utility_array.at[..., -1].set(
                jnp.clip(jnp.log(utility_array[..., -1]), -1e38)
            )
            return utility_array

        return u_rollup


    def __probability_for_nest(self, slot):
        nest_code = self.graph.standard_sort[slot]
        child_slots = self.graph.successor_slots(nest_code)
        mu_name = self.graph.nodes[nest_code].get('parameter')
        if mu_name is None:
            mu_slot = -1
        else:
            mu_slot = self.pf.index.get_loc(mu_name)

        # @jit
        def probability_nest(params, utility_array, probability_array):
            mu = params[mu_slot] if mu_slot >= 0 else 1.0
            u_nest = utility_array[..., slot]

            def body(carry, child_slot):
                add_me = (
                        (jnp.clip(utility_array[..., child_slot], -1e33) - jnp.clip(u_nest, -1e33)) / mu
                )
                carry = carry.at[..., child_slot].set(add_me + carry[..., slot])
                return carry, None

            probability_array, _ = jax.lax.scan(body, probability_array, child_slots)
            return probability_array

        return probability_nest

    @jitmethod
    def _jax_log_probability(self, params, ca=None, co=None, av=None):
        n_alts = self.dataset.n_alts
        n_nodes = len(self.graph)
        utility_array = self._jax_utility(params, ca, co, av)
        # downshift to prevent over/underflow
        shifter = utility_array[:n_alts].max(axis=-1)
        if av is not None:
            utility_array = utility_array.at[:n_alts].add(
                jnp.where(
                    av[:n_alts],
                    -shifter,
                    0
                )
            )
        else:
            utility_array = utility_array.at[:n_alts].add(
                -shifter
            )
        utility_array = self.utility_for_nests(utility_array, params, av)
        logprobability = jnp.zeros_like(utility_array)
        for slot in range(n_nodes, n_alts, -1):
            logprobability = self.__probability_for_nest(slot - 1)(params, utility_array, logprobability)
        return logprobability


    @jitmethod
    def jax_log_probability(self, params):
        n_alts = self.dataset.n_alts
        n_nodes = len(self.graph)
        utility_array = self.jax_utility(params)
        av = _get_jnp_array(self.dataset, "av")

        # downshift to prevent over/underflow
        shifter = utility_array[..., :n_alts].max(axis=-1)
        if av is not None:
            utility_array = utility_array.at[..., :n_alts].add(
                jnp.where(
                    av[..., :n_alts],
                    -shifter[..., None],
                    0
                )
            )
        else:
            utility_array = utility_array.at[..., :n_alts].add(
                -shifter[..., None]
            )
        utility_array = self.utility_for_nests(utility_array, params, av)
        logprobability = jnp.zeros_like(utility_array)
        for slot in range(n_nodes, n_alts, -1):
            logprobability = self.__probability_for_nest(slot - 1)(params, utility_array, logprobability)
        #return jnp.exp(logprobability[..., :n_alts])
        return logprobability

    @jitmethod
    def _jax_probability(self, params, ca=None, co=None, av=None):
        n_alts = self.dataset.n_alts
        return jnp.exp(self._jax_log_probability(params, ca, co, av)[:n_alts])

    @jitmethod
    def jax_probability(self, params):
        n_alts = self.dataset.n_alts
        return jnp.exp(self.jax_log_probability(params)[..., :n_alts])

    @compiledmethod
    def _jax_loglike(self):
        if len(self._mixtures) == 0:
            @jax.jit
            def loglike(params, ca=None, co=None, av=None, ch=None):
                # ch = jnp.asarray(self.dataset['ch'])
                logpr = self._jax_log_probability(params, ca, co, av)
                return (logpr[:ch.size] * ch).sum()
        else:
            raise NotImplementedError
            # if self._draws is None:
            #     self._make_random_draws(n_draws=self.n_draws)
            # @jax.jit
            # def loglike(params, ca=None, co=None, av=None, ch=None):
            #     pr = jax.vmap(self.jax_probability)(
            #         self.apply_random_draws(params)
            #     )
            #     likelihood = jnp.where(ch, pr, 1.0)
            #     if self.groupid is not None:
            #         in_group_likelihood = likelihood.prod(2).mean(0)
            #         in_group_log_likelihood = jnp.log(in_group_likelihood)
            #         return in_group_log_likelihood.sum()
            #     else:
            #         return jnp.log(likelihood.mean(0)).sum()
        return loglike

    @compiledmethod
    def jax_loglike_casewise(self):
        if len(self._mixtures) == 0:
            @jax.jit
            def loglike_casewise(params):
                ca = jnp.asarray(self.dataset['ca'])
                co = jnp.asarray(self.dataset['co'])
                av = jnp.asarray(self.dataset['av'])
                ch = jnp.asarray(self.dataset['ch'])
                f = jax.vmap(self._jax_loglike, in_axes=(
                    None,
                    None if ca is None else 0,
                    None if co is None else 0,
                    None if av is None else 0,
                    None if ch is None else 0,
                ), out_axes=0)
                return f(params, ca, co, av, ch)
        else:
            raise NotImplementedError
        return loglike_casewise

    @compiledmethod
    def jax_loglike(self):
        if len(self._mixtures) == 0:
            @jax.jit
            def loglike(params):
                return self.jax_loglike_casewise(params).sum()
        else:
            if self._draws is None:
                self._make_random_draws(n_draws=self.n_draws)
            @jax.jit
            def loglike(params):
                ch = jnp.asarray(self.dataset['ch'])
                n_alts = self.dataset.n_alts
                av = _get_jnp_array(self.dataset, "av")
                pr = jax.vmap(self.jax_probability)(
                    self.apply_random_draws(params)
                )
                likelihood = jnp.where(ch, pr, 1.0)
                if self.groupid is not None:
                    in_group_likelihood = likelihood.prod(2).mean(0)
                    in_group_log_likelihood = jnp.log(in_group_likelihood)
                    return in_group_log_likelihood.sum()
                else:
                    return jnp.log(likelihood.mean(0)).sum()
                # logpr = jnp.log(jnp.where(
                #     av[..., :n_alts],
                #     jnp.clip(pr, 1e-35),
                #     1.0
                # ))
                #return (logpr[:,:n_alts] * ch[:,:n_alts]).sum()
        return loglike

    @compiledmethod
    def jax_d_loglike(self):
        return jax.grad(self.jax_loglike)
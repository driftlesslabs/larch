import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from xarray import DataArray, Dataset

from ..compiled import compiledmethod, jitmethod, reset_compiled_methods
from ..folding import fold_dataset
from ..optimize import OptimizeMixin
from .numbamodel import NumbaModel
from .param_core import ParameterBucket


def _get_jnp_array(dataset, name):
    if name not in dataset:
        return None
    return jnp.asarray(dataset[name])


class PanelMixin:
    def __init__(self, *args, **kwargs):
        self._groupid = kwargs.pop("groupid", None)

    @property
    def groupid(self):
        return self._groupid

    @groupid.setter
    def groupid(self, g):
        if g is None or isinstance(g, str):
            if self._groupid != g:
                self.mangle()
        else:
            self.mangle()
        self._groupid = g


class Model(NumbaModel, OptimizeMixin, PanelMixin):
    def __init__(self, *args, **kwargs):
        PanelMixin.__init__(self, *args, **kwargs)
        super().__init__(*args, **kwargs)
        self._n_draws = 100
        self._draws = None
        self.prerolled_draws = True
        self.common_draws = False

    @property
    def compute_engine(self):
        engine = self._compute_engine
        if engine is None:
            engine = 'jax'
        return engine

    @compute_engine.setter
    def compute_engine(self, engine):
        if engine not in {'numba', 'jax', None}:
            raise ValueError('invalid compute engine')
        self._compute_engine = engine

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
    def pstderr(self):
        self.unmangle()
        return self._parameter_bucket.pstderr

    @pstderr.setter
    def pstderr(self, x):
        self._parameter_bucket.pstderr = x

    def mangle(self):
        super().mangle()
        self._draws = None
        reset_compiled_methods(self)

    def unmangle(self, force=False):
        if not self._mangled and not force:
            return
        marker = f"_currently_unmangling_{__file__}"
        if getattr(self, marker, False):
            return
        try:
            setattr(self, marker, True)
            super().unmangle(force=force)
            for mix in self.mixtures:
                mix.prep(self._parameter_bucket)
            if self.groupid is not None:
                self.dataset = fold_dataset(self.dataset, self.groupid)
        finally:
            delattr(self, marker)

    def reflow_data_arrays(self):
        """
        Reload the internal data_arrays so they are consistent with the datatree.
        """
        if self.graph is None:
            self._data_arrays = None
            return

        datatree = self.datatree
        if datatree is not None:

            request = self.required_data()
            if isinstance(self.groupid, str):
                request["group_co"] = self.groupid

            from .data_arrays import prepare_data

            dataset, self.dataflows = prepare_data(
                datasource=datatree,
                request=request,
                float_dtype=self.float_dtype,
                cache_dir=datatree.cache_dir,
                flows=getattr(self, "dataflows", None),
            )
            if isinstance(self.groupid, str):
                dataset = fold_dataset(dataset, "group")
            elif self.groupid is not None:
                dataset = fold_dataset(dataset, self.groupid)
            self.dataset = dataset
            try:
                self._data_arrays = self.dataset.dc.to_arrays(
                    self.graph, float_dtype=self.float_dtype,
                )
            except KeyError:  # no defined caseid dimension, JAX only
                self._data_arrays = None
                self.work_arrays = None
            else:
                if self.work_arrays is not None:
                    self._rebuild_work_arrays()

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

    def make_random_draws(self, engine='numpy'):
        self.unmangle()
        for i in self.mixtures:
            i.prep(self._parameter_bucket)
        n_panels = self.dataset.dc.n_panels
        n_mixtures = len(self.mixtures)
        n_draws = self.n_draws
        draws = None
        if self._draws is not None:
            if self.common_draws and self._draws.shape == (n_draws, n_mixtures):
                draws = self._draws
            if not self.common_draws and self._draws.shape == (n_panels, n_draws, n_mixtures):
                draws = self._draws
        if draws is None:
            if engine == 'numpy':
                seed = getattr(self, "seed", 0)
                if self.common_draws:
                    if n_draws > 0 and n_mixtures > 0:
                        draws, seed = self._make_random_draws_numpy(n_draws, n_mixtures, seed)
                else:
                    if n_draws > 0 and n_mixtures > 0 and n_panels > 0:
                        draws, seed = self._make_random_draws_numpy_2(
                            n_draws, n_mixtures, n_panels, seed
                        )
                    else:
                        draws = None
            elif engine == 'jax':
                seed = getattr(self, "seed", 0)
                rk = jax.random.PRNGKey(seed)
                if self.common_draws:
                    if n_draws > 0 and n_mixtures > 0:
                        draws = self._make_random_draws_out(n_draws, n_mixtures, rk)[
                            0
                        ]
                else:
                    if n_draws > 0 and n_mixtures > 0 and n_panels > 0:
                        draws = self._make_random_draws_out_2(
                            n_draws, n_mixtures, n_panels, rk
                        )[0]
                    else:
                        draws = None
            else:
                raise ValueError(f"unknown random engine {engine!r}")
            if self.prerolled_draws:
                self._draws = draws
        return draws

    def _make_random_draws_numpy(self, n_draws, n_mixtures, seed):
        if isinstance(seed, np.random.Generator):
            rgen = seed
        else:
            rgen = np.random.default_rng(seed)
        draws = rgen.random(size=[n_draws, n_mixtures], dtype=np.float32) + np.arange(n_draws, dtype=np.float32)[:, np.newaxis]
        for i in range(n_mixtures):
            rgen.shuffle(draws[:, i])
        return np.clip(draws / n_draws, 0, np.float32(1-1e-7)), rgen

    def _make_random_draws_numpy_2(self, n_draws, n_mixtures, n_panels, seed):
        if isinstance(seed, np.random.Generator):
            rgen = seed
        else:
            rgen = np.random.default_rng(seed)
        draws = rgen.random(size=[n_panels, n_draws, n_mixtures], dtype=np.float32) + np.arange(n_draws, dtype=np.float32)[np.newaxis, :, np.newaxis]
        for i in range(n_mixtures):
            for p in range(n_panels):
                rgen.shuffle(draws[p, :, i])
        return np.clip(draws / n_draws, 0, np.float32(1-1e-7)), rgen

    # @jitmethod(static_argnums=(0,1), static_argnames=('n_draws', 'n_mixtures'))
    def _make_random_draws_out(self, n_draws, n_mixtures, random_key):
        draws = jax.random.uniform(random_key, [n_draws, n_mixtures],)

        def body(i, carry):
            draws, rnd_key = carry
            rnd_key, subkey = jax.random.split(rnd_key)
            draws = draws.at[:, i].add(jax.random.permutation(subkey, draws.shape[0]))
            return draws, rnd_key

        draws, random_key = jax.lax.fori_loop(
            0, draws.shape[1], body, (draws, random_key)
        )
        return draws * (1 / n_draws), random_key

    # @jitmethod(static_argnums=(0,1,2), static_argnames=('n_draws', 'n_mixtures', 'n_cases'))
    def _make_random_draws_out_2(self, n_draws, n_mixtures, n_cases, random_key):
        def body(carry, x):
            rkey = carry
            draws, rkey = self._make_random_draws_out(n_draws, n_mixtures, rkey)
            return rkey, draws

        random_key, draws = jax.lax.scan(body, random_key, None, n_cases)
        return draws, random_key

    def apply_random_draws(self, parameters, draws=None):
        # if draws is None:
        #     draws = self._draws
        parameters = jnp.broadcast_to(
            parameters, [*draws.shape[:-1], parameters.shape[0]]
        )
        for mix_n, mix in enumerate(self.mixtures):
            u = draws[..., mix_n]
            parameters = mix.roll(u, parameters)
        return parameters

    @jitmethod
    def _jax_utility(self, params, databundle):
        ca = databundle.get("ca", None)
        co = databundle.get("co", None)
        av = databundle.get("av", None)
        if co is None:
            n_vars_co = 0
        else:
            n_vars_co = co.shape[-1]
        n_alts = self.dataset.dc.n_alts
        n_nodes = len(self.graph)
        x = jnp.zeros([self.dataset.dc.n_alts, n_vars_co + 1])
        x = x.at[self._fixed_arrays.uco_alt_slot, self._fixed_arrays.uco_data_slot].add(
            params[self._fixed_arrays.uco_param_slot]
        )
        u = jnp.zeros([n_nodes])
        if ca is not None and self._fixed_arrays.uca_param_slot.size:
            u = u.at[:n_alts].add(
                jnp.dot(
                    ca[..., self._fixed_arrays.uca_data_slot],
                    params[self._fixed_arrays.uca_param_slot],
                )
            )
        if co is not None:
            temp = jnp.dot(co, x[:, :-1].T)
            u = u.at[:n_alts].add(temp)
        u = u.at[:n_alts].add(x[:, -1].T)
        if av is not None and not self.availability_any:
            u = u.at[:n_alts].set(jnp.where(av[:n_alts], u[:n_alts], -jnp.inf))
        return u

    @jitmethod
    def jax_utility(self, params):
        ca = _get_jnp_array(self.dataset, "ca")
        co = _get_jnp_array(self.dataset, "co")
        av = _get_jnp_array(self.dataset, "av")
        if av is not None:
            depth = av.ndim - 1
        elif co is not None:
            depth = co.ndim - 1
        elif ca is not None:
            depth = ca.ndim - 2
        else:
            raise ValueError("missing data")
        f = self._jax_utility
        for level in range(depth):
            f = jax.vmap(f, in_axes=(None, 0))
        return f(params, {"ca": ca, "co": co, "av": av})

    def __utility_for_nest(self, slot):
        nest_code = self.graph.standard_sort[slot]
        child_slots = self.graph.successor_slots(nest_code)
        mu_name = self.graph.nodes[nest_code].get("parameter")
        if mu_name is None:
            mu_slot = -1
        else:
            mu_slot = self.get_param_loc(mu_name)

        # @jit
        def u_nest(params, utility_array, array_av):
            mu = params[mu_slot] if mu_slot >= 0 else 1.0
            carry = jnp.zeros(utility_array.shape[:-1])
            # num_av = jnp.sum(utility_array[..., child_slots] > -1e30, axis=-1)
            num_av = array_av[..., slot]

            def body(carry, child_slot):
                carry = jnp.add(
                    carry,
                    jnp.where(
                        num_av > 1,
                        jnp.exp(jnp.clip(utility_array[..., child_slot], -1e37) / mu),
                        jnp.exp(jnp.clip(utility_array[..., child_slot], -1e37)),
                    ),
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
            return out.at[..., -1].set(jnp.log(jnp.exp(out[..., :-1]).sum(-1)))

        if n_nodes - n_alts <= 1:
            return u_nesting_none

        # @jit
        def u_nesting_few(out, beta, array_av):
            for slot in range(n_alts, n_nodes):
                out = self.__utility_for_nest(slot)(beta, out, array_av)
            return out

        if n_nodes - n_alts < 3:
            return u_nesting_few

        # many nests, use more efficient loop
        n_params = self.n_params
        mu_names = [
            self.graph.nodes[i].get("parameter") for i in self.graph.standard_sort
        ]
        mu_slots = jnp.asarray(
            [
                (self.get_param_loc(mu_name) if mu_name is not None else n_params)
                for mu_name in mu_names
            ]
        )

        def add_u_to_parent(u, params, child_slot, parent_slot, avail):
            mu = params[mu_slots[parent_slot]]
            u = u.at[..., parent_slot].add(
                jnp.where(
                    avail[..., parent_slot] > 1,
                    jnp.exp(jnp.clip(u[..., child_slot], -1e37) / mu),
                    jnp.exp(jnp.clip(u[..., child_slot], -1e37)),
                )
            )
            return u

        def log_self(u, params, self_slot, avail):
            mu = params[mu_slots[self_slot]]
            u = u.at[..., self_slot].set(
                jnp.where(
                    avail[..., self_slot] > 1,
                    jnp.clip(jnp.log(u[..., self_slot]), -1e38) * mu,
                    jnp.clip(jnp.log(u[..., self_slot]), -1e38),
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
                    (firstvisit >= 0) & (dn_slot >= n_alts),
                    lambda u: log_self(u, params, dn_slot, avail_ca),
                    lambda u: u,
                    operand=u,
                )
                u = add_u_to_parent(u, params, dn_slot, up_slot, avail_ca)
                return (u, params), None

            (utility_array, _ignore_1), _ignore_2 = jax.lax.scan(
                body, (utility_array, params), slotarray
            )
            # log utility at root
            utility_array = utility_array.at[..., -1].set(
                jnp.clip(jnp.log(utility_array[..., -1]), -1e38)
            )
            return utility_array

        return u_rollup

    def __probability_for_nest(self, slot):
        nest_code = self.graph.standard_sort[slot]
        child_slots = self.graph.successor_slots(nest_code)
        mu_name = self.graph.nodes[nest_code].get("parameter")
        if mu_name is None:
            mu_slot = -1
        else:
            mu_slot = self.get_param_loc(mu_name)

        # @jit
        def probability_nest(params, utility_array, probability_array):
            mu = params[mu_slot] if mu_slot >= 0 else 1.0
            u_nest = utility_array[..., slot]

            def body(carry, child_slot):
                add_me = (
                    jnp.clip(utility_array[..., child_slot], -1e33)
                    - jnp.clip(u_nest, -1e33)
                ) / mu
                carry = carry.at[..., child_slot].set(add_me + carry[..., slot])
                return carry, None

            probability_array, _ = jax.lax.scan(body, probability_array, child_slots)
            return probability_array

        return probability_nest

    @jitmethod
    def _jax_log_probability_bundle(self, params, databundle):
        ca, co, ch, av, wt = databundle
        return self._jax_log_probability(params, ca, co, av)

    @jitmethod
    def _jax_log_probability(self, params, databundle):
        av = databundle.get("av", None)
        n_alts = self.dataset.dc.n_alts
        n_nodes = len(self.graph)
        utility_array = self._jax_utility(params, databundle)
        # downshift to prevent over/underflow
        shifter = utility_array[:n_alts].max(axis=-1)
        if av is not None and not self.availability_any:
            utility_array = utility_array.at[:n_alts].add(
                jnp.where(av[:n_alts], -shifter, 0)
            )
        else:
            utility_array = utility_array.at[:n_alts].add(-shifter)
        utility_array = self.utility_for_nests(utility_array, params, av)
        logprobability = jnp.zeros_like(utility_array)
        for slot in range(n_nodes, n_alts, -1):
            logprobability = self.__probability_for_nest(slot - 1)(
                params, utility_array, logprobability
            )
        return logprobability

    @jitmethod
    def jax_log_probability(self, params):
        ca = _get_jnp_array(self.dataset, "ca")
        co = _get_jnp_array(self.dataset, "co")
        av = _get_jnp_array(self.dataset, "av")
        if av is not None:
            depth = av.ndim - 1
        elif co is not None:
            depth = co.ndim - 1
        elif ca is not None:
            depth = ca.ndim - 2
        else:
            raise ValueError("missing data")
        f = self._jax_log_probability
        for level in range(depth):
            f = jax.vmap(f, in_axes=(None, 0))
        return f(params, {"ca": ca, "co": co, "av": av})

    @jitmethod
    def _jax_probability(self, params, databundle):
        n_alts = self.dataset.dc.n_alts
        return jnp.exp(self._jax_log_probability(params, databundle)[:n_alts])

    @jitmethod
    def jax_probability(self, params):
        ca = _get_jnp_array(self.dataset, "ca")
        co = _get_jnp_array(self.dataset, "co")
        av = _get_jnp_array(self.dataset, "av")
        if av is not None:
            depth = av.ndim - 1
        elif co is not None:
            depth = co.ndim - 1
        elif ca is not None:
            depth = ca.ndim - 2
        else:
            raise ValueError("missing data")
        f = self._jax_probability
        for level in range(depth):
            f = jax.vmap(f, in_axes=(None, 0))
        return f(params, {"ca": ca, "co": co, "av": av})

    @jitmethod
    def _jax_likelihood(self, params, databundle):
        n_alts = self.dataset.dc.n_alts
        ch = databundle.get("ch", None)[:n_alts]
        pr = self._jax_probability(params, databundle)[:n_alts]
        likely = jnp.where(ch, pr, 1.0)  # TODO make power if needed
        return likely

    @jitmethod(static_argnums=(3,), static_argnames="n_draws")
    def _jax_loglike_casewise(self, params, databundle, groupbundle=None, n_draws=100):
        if len(self.mixtures) == 0:
            logpr = self._jax_log_probability(params, databundle)
            ch = databundle.get("ch", None)
            return (logpr[: ch.size] * ch).sum()
        else:
            if self.prerolled_draws:
                draws = groupbundle.get("draws", None)
            else:
                rk = groupbundle.get("rk", None)
                draws, _ = self._make_random_draws_out(n_draws, len(self.mixtures), rk)
            ch = databundle.get("ch", None)
            rand_params = self.apply_random_draws(params, draws)
            if ch.ndim == 2:  # PANEL DATA
                # vmap over ingroup
                likelihood_f = jax.vmap(self._jax_likelihood, in_axes=(None, 0),)
                # vmap over draws
                likelihood = jax.vmap(likelihood_f, in_axes=(0, None), out_axes=-1,)(
                    rand_params, databundle
                )
                # collapse likelihood over all alternatives
                likelihood = likelihood.prod([0, 1])
                # average over all draws
                likelihood = likelihood.mean(0)
                return jnp.log(likelihood)
            else:
                # vmap over draws
                likelihood = jax.vmap(
                    self._jax_likelihood, in_axes=(0, None), out_axes=-1,
                )(rand_params, databundle)
                # collapse likelihood over all alternatives
                likelihood = likelihood.prod(0)
                # average over all draws
                likelihood = likelihood.mean(0)
                return jnp.log(likelihood)

    @jitmethod
    def jax_loglike_casewise(self, params):
        ca = _get_jnp_array(self.dataset, "ca")
        co = _get_jnp_array(self.dataset, "co")
        av = _get_jnp_array(self.dataset, "av")
        ch = _get_jnp_array(self.dataset, "ch")
        n_draws = self.n_draws
        seed = getattr(self, "seed", 42)
        if av is not None:
            depth = av.ndim - 1
            shape = av.shape[:-1]
        elif co is not None:
            depth = co.ndim - 1
            shape = co.shape[:-1]
        elif ca is not None:
            depth = ca.ndim - 2
            shape = ca.shape[:-2]
        elif ch is not None:
            depth = ch.ndim - 1
            shape = ch.shape[:-1]
        else:
            raise ValueError("missing data")
        random_key = jax.random.PRNGKey(seed)
        if self.groupid is not None:
            depth = depth - 1
            shape = shape[:-1]
        f = (
            self._jax_loglike_casewise
        )  # params, databundle, groupbundle=None, n_draws=100
        from .random import keysplit

        commons = None if self.common_draws else 0
        for i in range(depth):
            f = jax.vmap(f, in_axes=(None, 0, commons, None))
            if not self.prerolled_draws:
                random_key, shape = keysplit(random_key, shape)
        if self.prerolled_draws:
            return f(
                params,
                {"ca": ca, "co": co, "av": av, "ch": ch},
                {"draws": self._draws},
                n_draws,
            )
        else:
            return f(
                params,
                {"ca": ca, "co": co, "av": av, "ch": ch},
                {"rk": random_key},
                n_draws,
            )

    @jitmethod
    def jax_loglike(self, params):
        return self.jax_loglike_casewise(params).sum()

    def loglike(
            self,
            x=None,
            *,
            start_case=None, stop_case=None, step_case=None,
            **kwargs
    ):
        if self.compute_engine != 'jax':
            return super().loglike(x=x, start_case=start_case, stop_case=stop_case, step_case=step_case, **kwargs)
        if start_case is not None:
            raise NotImplementedError('start_case with engine=jax')
        if stop_case is not None:
            raise NotImplementedError('stop_case with engine=jax')
        if step_case is not None:
            raise NotImplementedError('step_case with engine=jax')
        # if kwargs:
        #     raise NotImplementedError(f"{kwargs.popitem()[0]} with engine=jax")
        if x is not None:
            self.pvals = x
        result = float(self.jax_loglike(self.pvals))
        if start_case is None and stop_case is None and step_case is None:
            self._check_if_best(result)
        return result

    def d_loglike(
            self,
            x=None,
            *,
            start_case=None, stop_case=None, step_case=None,
            return_series=False,
            **kwargs,
    ):
        if self.compute_engine != 'jax':
            return super().d_loglike(x=x, start_case=start_case, stop_case=stop_case, step_case=step_case, return_series=return_series, **kwargs)
        if start_case is not None:
            raise NotImplementedError('start_case with engine=jax')
        if stop_case is not None:
            raise NotImplementedError('stop_case with engine=jax')
        if step_case is not None:
            raise NotImplementedError('step_case with engine=jax')
        # if kwargs:
        #     raise NotImplementedError(f"{kwargs.popitem()[0]} with engine=jax")
        if x is not None:
            self.pvals = x
        result = self.jax_d_loglike(self.pvals)

        print("converge?=", jnp.max(jnp.absolute(result)))

        if return_series:
            result = pd.Series(result, index=self.pnames)
        return result


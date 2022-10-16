import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr

from ..compiled import compiledmethod, jitmethod, reset_compiled_methods
from ..dataset import Dataset, DataTree
from ..folding import fold_dataset
from ..optimize import OptimizeMixin
from .jaxmodel import PanelMixin, _get_jnp_array
from .mixtures import MixtureList
from .param_core import ParameterBucket


class LatentClass(ParameterBucket, OptimizeMixin, PanelMixin):
    compute_engine = "jax"

    def __init__(
        self, classmodel, choicemodels, datatree=None, float_dtype=np.float32, **kwargs
    ):
        PanelMixin.__init__(self, **kwargs)
        self._is_mangled = True
        self._dataset = None
        super().__init__(choicemodels, classmodel=classmodel)
        self.datatree = datatree
        if classmodel.dataset is not None:
            assert sorted(classmodel.dataset.dc.altids()) == sorted(choicemodels.keys())
        for choicemodel in choicemodels.values():
            self.dataset = choicemodel.dataset
        if float_dtype is not None:
            for v in self._models.values():
                v.float_dtype = float_dtype

    def save(self, filename, format="yaml", overwrite=False):
        from .saving import save_model

        return save_model(self, filename, format=format, overwrite=overwrite)

    @classmethod
    def from_dict(cls, content):
        _models = content.get("_models")
        from .saving import load_model

        classmodel = None
        choicemodels = {}
        for k, v in _models.items():
            if k == "classmodel":
                classmodel = load_model(v)
            else:
                choicemodels[k] = load_model(v)
        self = cls(classmodel, choicemodels)

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
        loadthis("constraint_intensity")
        loadthis("constraint_sharpness")
        loadthis("constraints")
        from .tree import NestingTree

        loadthis("graph", NestingTree.from_dict)
        loadthis("groupid")
        loadthis("logsum_parameter")
        loadthis("quantity_ca")
        loadthis("quantity_scale")
        loadthis("title")
        loadthis("utility_ca")
        loadthis("utility_co")
        loadthis("weight_co_var")
        loadthis("weight_normalization")
        return self

    @property
    def pf(self):
        return self.parameters.to_dataframe()

    @jitmethod
    def jax_probability(self, params):
        classmodel = self["classmodel"]
        pr_parts = []
        for n, k in enumerate(self["classmodel"].dataset.dc.altids()):
            pr_parts.append(
                self._models[k].jax_probability(params)
                * jnp.expand_dims(classmodel.jax_probability(params)[..., n], -1)
            )
        return sum(pr_parts)

    @jitmethod
    def jax_loglike_casewise(self, params):
        n_alts = self.dataset.dc.n_alts
        ch = jnp.asarray(self.dataset["ch"])
        if ch.ndim == 2:
            pr = self.jax_probability(params)
            masked_pr = jnp.where(ch[..., :n_alts] > 0, pr[..., :n_alts], 1.0)
            log_pr = jnp.log(masked_pr)
            return (log_pr[..., :n_alts] * ch[..., :n_alts]).sum()
        elif ch.ndim >= 3:
            classmodel = self["classmodel"]
            likely_parts = []
            for n, k in enumerate(self["classmodel"].dataset.dc.altids()):
                k_pr = self._models[k].jax_probability(params)
                masked_k_pr = jnp.where(ch[..., :n_alts] > 0, k_pr[..., :n_alts], 1.0)
                k_likely = jnp.power(masked_k_pr, ch[..., :n_alts]).prod([-2, -1])
                likely_parts.append(
                    k_likely * classmodel.jax_probability(params)[..., 0, n]
                )
            return jnp.log(sum(likely_parts))

    @jitmethod
    def jax_loglike(self, params):
        return self.jax_loglike_casewise(params).sum()

    def loglike(
        self, x=None, *, start_case=None, stop_case=None, step_case=None, **kwargs
    ):
        if self.compute_engine != "jax":
            raise NotImplementedError(f"latent class with engine={self.compute_engine}")
        if start_case is not None:
            raise NotImplementedError("start_case with engine=jax")
        if stop_case is not None:
            raise NotImplementedError("stop_case with engine=jax")
        if step_case is not None:
            raise NotImplementedError("step_case with engine=jax")
        if x is not None:
            self.pvals = x
        result = float(self.jax_loglike(self.pvals))
        # if start_case is None and stop_case is None and step_case is None:
        #     self._check_if_best(result)
        return result

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

    @jitmethod
    def jax_d_loglike(self, params):
        return jax.grad(self.jax_loglike)(params)

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
        if self.compute_engine != "jax":
            raise NotImplementedError(f"latent class with engine={self.compute_engine}")
        if start_case is not None:
            raise NotImplementedError("start_case with engine=jax")
        if stop_case is not None:
            raise NotImplementedError("stop_case with engine=jax")
        if step_case is not None:
            raise NotImplementedError("step_case with engine=jax")
        if x is not None:
            self.pvals = x
        result = self.jax_d_loglike(self.pvals)

        print("converge?=", jnp.max(jnp.absolute(result)))

        if return_series:
            result = pd.Series(result, index=self.pnames)
        return result

    def neg_d_loglike(self, x=None, start_case=0, stop_case=-1, step_case=1, **kwargs):
        result = self.d_loglike(
            x, start_case=start_case, stop_case=stop_case, step_case=step_case, **kwargs
        )
        return -np.asarray(result)

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

    def reflow_data_arrays(self):
        """
        Reload the internal data_arrays so they are consistent with the datatree.
        """

        datatree = self.datatree
        if datatree is None:
            raise ValueError("missing datatree")

        request = self._models["classmodel"].required_data()
        request.pop("avail_any", None)
        for kname, kmodel in self._models.items():
            if kname == "classmodel":
                continue
            kreq = kmodel.required_data()
            for k, v in kreq.items():
                if k not in request:
                    request[k] = v
                else:
                    if isinstance(request[k], dict):
                        request[k].update(v)
                    elif isinstance(request[k], (list, tuple)):
                        request[k] = list(set(request[k]) | set(v))
                    else:
                        if request[k] != v:
                            raise ValueError("incompatible requests")

        if isinstance(self.groupid, str):
            request["group_co"] = self.groupid

        from .data_arrays import prepare_data

        dataset, self.dataflows = prepare_data(
            datasource=datatree,
            request=request,
            float_dtype=np.float32,
            cache_dir=datatree.cache_dir,
            flows=getattr(self, "dataflows", None),
        )
        if isinstance(self.groupid, str):
            dataset = fold_dataset(dataset, "group")
        elif self.groupid is not None:
            dataset = fold_dataset(dataset, self.groupid)
        self.dataset = dataset

        for kname, kmodel in self._models.items():
            if kname == "classmodel":
                continue
            kmodel.dataset = self.dataset
        self._models["classmodel"].dataset = self.dataset.drop_dims(
            self.dataset.dc.ALTID
        ).dc.set_altids(self._models["classmodel"].dataset.dc.altids())

    def mangle(self):
        super().mangle()
        reset_compiled_methods(self)
        self._is_mangled = True

    def unmangle(self, *args, **kwargs):
        if self._is_mangled:
            super().unmangle(*args, **kwargs)
            self.reflow_data_arrays()
            self._is_mangled = False

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
            self.mangle()
        elif isinstance(dataset, _Dataset):
            self._dataset = Dataset(dataset)
            self._data_arrays = None
            self.mangle()
        else:
            raise TypeError(f"dataset must be Dataset not {type(dataset)}")

    @property
    def datatree(self):
        """DataTree : A source for data for the model"""
        try:
            return self._datatree
        except AttributeError:
            return None

    @datatree.setter
    def datatree(self, tree):
        if tree is self.datatree:
            return
        if isinstance(tree, DataTree) or tree is None:
            self._datatree = tree
            self.mangle()
        elif isinstance(tree, Dataset):
            self._datatree = tree.dc.as_tree()
            self.mangle()
        else:
            try:
                self._datatree = DataTree(main=Dataset.construct(tree))
            except Exception as err:
                raise TypeError(f"datatree must be DataTree not {type(tree)}") from err
            else:
                self.mangle()

    @property
    def data_as_loaded(self):
        return self._dataset


class MixedLatentClass(LatentClass):

    mixtures = MixtureList()

    def __init__(
        self, *args, n_draws=100, prerolled_draws=True, common_draws=False, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._n_draws = n_draws
        self._draws = None
        self.prerolled_draws = prerolled_draws
        self.common_draws = common_draws

    @classmethod
    def from_dict(cls, content):
        self = super().from_dict(content)

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

        loadthis("mixtures", self.mixtures.from_list)
        loadthis("n_draws")
        loadthis("prerolled_draws")
        loadthis("common_draws")

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

    def _make_random_draws_out(self, n_draws, n_mixtures, random_key):
        draws = jax.random.uniform(
            random_key,
            [n_draws, n_mixtures],
        )

        def body(i, carry):
            draws, rnd_key = carry
            rnd_key, subkey = jax.random.split(rnd_key)
            draws = draws.at[:, i].add(jax.random.permutation(subkey, draws.shape[0]))
            return draws, rnd_key

        draws, random_key = jax.lax.fori_loop(
            0, draws.shape[1], body, (draws, random_key)
        )
        return draws * (1 / n_draws), random_key

    @jitmethod
    def _jax_probability_by_class(self, params, databundle):
        # classmodel = self['classmodel']
        pr_parts = []
        for n, k in enumerate(self["classmodel"].dataset.dc.altids()):
            pr_parts.append(
                self._models[k]._jax_probability(params, databundle)
                # * jnp.expand_dims(classmodel._jax_probability(params, databundle)[..., n], -1)
            )
        return jnp.stack(pr_parts)

    @jitmethod
    def _jax_likelihood_by_class(self, params, databundle):
        n_alts = self.dataset.dc.n_alts
        ch = databundle.get("ch", None)[..., :n_alts]
        pr_parts = self._jax_probability_by_class(params, databundle)[..., :n_alts]
        likely = jnp.where(
            jnp.expand_dims(ch, 0), pr_parts, 1.0
        )  # TODO make power if needed
        return likely

    @jitmethod(static_argnums=(3,), static_argnames="n_draws")
    def _jax_loglike_casewise_mixed(
        self, params, databundle, groupbundle=None, n_draws=100
    ):
        classmodel = self["classmodel"]
        if self.prerolled_draws:
            draws = groupbundle.get("draws", None)
        else:
            rk = groupbundle.get("rk", None)
            draws, _ = self._make_random_draws_out(n_draws, len(self.mixtures), rk)
        ch = databundle.get("ch", None)
        rand_params = self.apply_random_draws(params, draws)
        if ch.ndim == 2:  # PANEL DATA
            # vmap over ingroup
            likelihood_f = jax.vmap(
                self._jax_likelihood_by_class,
                in_axes=(None, 0),
            )
            # vmap over draws
            likelihood = jax.vmap(
                likelihood_f,
                in_axes=(0, None),
                out_axes=-1,
            )(rand_params, databundle)
            # collapse likelihood over all alternatives over all cases-within-panel
            likelihood = likelihood.prod([0, 2])
            # likelihood.shape is now (n_classes, n_draws)
            ################
            class_pr = jax.vmap(
                classmodel._jax_probability,
                in_axes=(0, None),
                out_axes=-1,
            )(
                rand_params,
                {
                    "co": databundle["co"][0],
                },
            )
            # class_pr.shape = (nclasses, ndraws)
            meta_likely = (likelihood * class_pr).sum(0).mean(0)
            return jnp.log(meta_likely)
        else:
            # vmap over draws
            likelihood = jax.vmap(
                self._jax_likelihood_by_class,
                in_axes=(0, None),
                out_axes=-1,
            )(rand_params, databundle)
            # collapse likelihood over all alternatives
            likelihood = likelihood.prod(0)
            # average over all draws
            likelihood = likelihood.mean(0)
            return jnp.log(likelihood)

    @jitmethod
    def jax_loglike_casewise(self, params):
        if len(self.mixtures) == 0:
            return super().jax_loglike_casewise(params)
        ca = _get_jnp_array(self.dataset, "ca")
        co = _get_jnp_array(self.dataset, "co")
        av = _get_jnp_array(self.dataset, "av")
        ch = _get_jnp_array(self.dataset, "ch")
        n_draws = self._n_draws
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
            self._jax_loglike_casewise_mixed
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

    def make_random_draws(self, engine="numpy"):
        self.unmangle()
        for i in self.mixtures:
            i.prep(self)
        n_panels = self.dataset.dc.n_panels
        n_mixtures = len(self.mixtures)
        n_draws = self._n_draws
        draws = None
        if self._draws is not None:
            if self.common_draws and self._draws.shape == (n_draws, n_mixtures):
                draws = self._draws
            if not self.common_draws and self._draws.shape == (
                n_panels,
                n_draws,
                n_mixtures,
            ):
                draws = self._draws
        if draws is None:
            if engine == "numpy":
                seed = getattr(self, "seed", 0)
                if self.common_draws:
                    if n_draws > 0 and n_mixtures > 0:
                        draws, seed = self._make_random_draws_numpy(
                            n_draws, n_mixtures, seed
                        )
                else:
                    if n_draws > 0 and n_mixtures > 0 and n_panels > 0:
                        draws, seed = self._make_random_draws_numpy_2(
                            n_draws, n_mixtures, n_panels, seed
                        )
                    else:
                        draws = None
            elif engine == "jax":
                seed = getattr(self, "seed", 0)
                rk = jax.random.PRNGKey(seed)
                if self.common_draws:
                    if n_draws > 0 and n_mixtures > 0:
                        draws = self._make_random_draws_out(n_draws, n_mixtures, rk)[0]
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

    # @jitmethod(static_argnums=(0,1,2), static_argnames=('n_draws', 'n_mixtures', 'n_cases'))
    def _make_random_draws_out_2(self, n_draws, n_mixtures, n_cases, random_key):
        def body(carry, x):
            rkey = carry
            draws, rkey = self._make_random_draws_out(n_draws, n_mixtures, rkey)
            return rkey, draws

        random_key, draws = jax.lax.scan(body, random_key, None, n_cases)
        return draws, random_key

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from .._optional import jax, jnp
from ..compiled import jitmethod, reset_compiled_methods
from ..dataset import Dataset, DataTree
from ..exceptions import MissingDataError
from ..folding import dissolve_zero_variance, fold_dataset
from ..optimize import OptimizeMixin
from ..util.simple_attribute import SimpleAttribute
from .basemodel import MANGLE_DATA, BaseModel
from .jaxmodel import PanelMixin, _get_jnp_array
from .mixtures import MixtureList


class LatentClass(BaseModel, OptimizeMixin, PanelMixin):
    compute_engine = "jax"
    float_dtype = SimpleAttribute()
    dataflows = SimpleAttribute(dict)

    def __init__(
        self,
        classmodel: BaseModel,
        choicemodels: dict[int, BaseModel],
        datatree: DataTree | None = None,
        float_dtype: type = np.float32,
        **kwargs,
    ):
        """
        Initialize the latent class model structure.

        This structure is used to connect two or more discrete classes, as well as
        a class membership model that determines the probability of each decision
        maker being a member of each class.

        The estimation of latent class models cannot be done with the
        `numba` compute engine, and must be done with the `jax` compute engine.

        Parameters
        ----------
        classmodel : BaseModel
            The class membership model.
        choicemodels : dict
            A dictionary of choice models.  The keys of this dictionary will be
            the altids of the choice models used in the class membership model.
        datatree : DataTree, optional
            A DataTree to use for all submodels.  For the choice models, this will
            override any existing datatree set on those models.  For the class
            membership model, this will be used to set the datatree only if it is
            not already set, as it may be desirable in some situations to have the
            class membership model use a different datatree than the choice models.
        float_dtype : type, optional
            The float type to use for the model.  Defaults to np.float32.
        """
        self._model_subtype = "latent-class"
        classmodel._model_subtype = "class-membership"
        for k, m in choicemodels.items():
            m.ident = k
        PanelMixin.__init__(self, **kwargs)
        self._is_mangled = True
        self._dataset = None
        choicemodels_keys = sorted(choicemodels.keys())
        if (
            classmodel.datatree is None
            and isinstance(datatree, Dataset)
            and self.groupid
        ):
            # We have not assigned data to the class model yet, but we have been
            # given a datatree and a groupid, so we will use that to create the
            # necessary data for the class model.
            df = datatree.to_dataframe()
            df["ingroup"] = df.groupby(self.groupid).cumcount() + 1
            classdata = dissolve_zero_variance(
                df.set_index([self.groupid, "ingroup"], drop=True).to_xarray(),
                "ingroup",
            )
            classdata = classdata.dc.set_altids(choicemodels_keys).drop_dims("ingroup")
            classdata.dc.CASEID = self.groupid
            classmodel.datatree = classdata
        elif classmodel.datatree is None and isinstance(datatree, Dataset):
            # The class model has not been assigned data yet, but there is no
            # groupid, so we will simply assign the datatree to the class model.
            # We still need to set the altids to match the choicemodels.
            classmodel.datatree = datatree.dc.set_altids(choicemodels_keys)
        else:
            pass
        self._ident = "latent-class"
        super().__init__(
            datatree=datatree,
            submodels=choicemodels,
            named_submodels={"classmodel": classmodel},
        )
        # self.datatree = datatree
        if classmodel.dataset is not None:
            assert sorted(classmodel.dataset.dc.altids()) == choicemodels_keys
        if float_dtype is not None:
            for v in self._models.values():
                v.float_dtype = float_dtype
        for _k, m in self._models.items():
            if m._model_subtype == "latent-class":
                pass
            elif m._model_subtype == "class-membership":
                m.groupid = self.groupid
            else:
                m.datatree = self.datatree
                m.groupid = self.groupid

    @property
    def _models(self):
        return self._parameter_bucket._models

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
            if v._model_subtype == "class-membership":
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
        classmodel = self.class_membership_model
        pr_parts = []
        for n, k in enumerate(classmodel.dataset.dc.altids()):
            class_pr = jnp.expand_dims(classmodel.jax_probability(params)[..., n], -1)
            inclass_pr = self._models[k].jax_probability(params)
            if inclass_pr.ndim < class_pr.ndim:
                inclass_pr = inclass_pr.reshape(
                    class_pr.shape[0],
                    inclass_pr.shape[0] // class_pr.shape[0],
                    *inclass_pr.shape[1:],
                )
            pr_parts.append(inclass_pr * class_pr)
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
            classmodel = self.class_membership_model
            class_pr = classmodel.jax_probability(params)
            likely_parts = []
            for n, k in enumerate(classmodel.dataset.dc.altids()):
                _yo_data = self._models[k].dataset
                k_pr = self._models[k].jax_probability(params)  # .reshape(ch.shape)
                masked_k_pr = jnp.where(ch[..., :n_alts] > 0, k_pr[..., :n_alts], 1.0)
                k_likely = jnp.power(masked_k_pr, ch[..., :n_alts]).prod([-2, -1])
                likely_parts.append(k_likely * class_pr[..., 0, n])
            return jnp.log(sum(likely_parts))

    @jitmethod
    def jax_loglike(self, params):
        return self.jax_loglike_casewise(params).sum()

    def loglike(
        self,
        x=None,
        *,
        start_case=None,
        stop_case=None,
        step_case=None,
        check_if_best=True,
        **kwargs,
    ):
        if self.compute_engine != "jax":
            raise NotImplementedError(f"latent class with engine={self.compute_engine}")
        self.unmangle()
        if start_case is not None and start_case != 0:
            raise NotImplementedError(f"{start_case=} with engine=jax")
        if stop_case is not None and stop_case != -1:
            raise NotImplementedError(f"{stop_case=} with engine=jax")
        if step_case is not None and step_case != 1:
            raise NotImplementedError(f"{step_case=} with engine=jax")
        if x is not None:
            self.pvals = x
        result = float(self.jax_loglike(self.pvals))
        if (
            check_if_best
            and start_case is None
            and stop_case is None
            and step_case is None
        ):
            self._check_if_best(result)
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
        self.unmangle()
        if start_case is not None and start_case != 0:
            raise NotImplementedError(f"{start_case=} with engine=jax")
        if stop_case is not None and stop_case != -1:
            raise NotImplementedError(f"{stop_case=} with engine=jax")
        if step_case is not None and step_case != 1:
            raise NotImplementedError(f"{step_case=} with engine=jax")
        if x is not None:
            self.pvals = x
        result = self.jax_d_loglike(self.pvals)

        # print("converge?=", jnp.max(jnp.absolute(result)))

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
        """Reload the internal data_arrays so they are consistent with the datatree."""
        datatree = self.datatree
        if datatree is None:
            raise ValueError("missing datatree")

        classmodel = self.class_membership_model
        request = classmodel.required_data()
        request.pop("avail_any", None)
        for _kname, kmodel in self._models.items():
            if kmodel._model_subtype in ("class-membership", "latent-class"):
                continue
            kreq = kmodel.required_data()
            for k, v in kreq.items():
                if k not in request:
                    request[k] = v
                else:
                    if isinstance(request[k], dict):
                        request[k].update(v)
                    elif isinstance(request[k], (list | tuple)):
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
            flows=self.dataflows,
        )
        if isinstance(self.groupid, str):
            dataset = fold_dataset(dataset, "group")
        elif self.groupid is not None:
            dataset = fold_dataset(dataset, self.groupid)
        self.dataset = dataset

        for _kname, kmodel in self._models.items():
            if kmodel._model_subtype in ("class-membership", "latent-class"):
                continue
            kmodel.dataset = self.dataset
            # kmodel.reflow_data_arrays() # not full reflow, just...
            kmodel._data_arrays = kmodel.dataset.dc.to_arrays(
                kmodel.graph,
                float_dtype=kmodel.float_dtype,
            )
            # if kmodel.work_arrays is not None:
            kmodel._rebuild_fixed_arrays()
            kmodel._rebuild_work_arrays()
        classmodel_ids = [
            kid
            for (kid, km) in self._models.items()
            if km._model_subtype not in ("latent-class", "class-membership")
        ]
        classmodel_data = dissolve_zero_variance(
            self.dataset.drop_dims(self.dataset.dc.ALTID).dc.set_altids(classmodel_ids),
            "ingroup",
        )
        classmodel.dataset = classmodel_data
        classmodel.reflow_data_arrays()
        # classmodel._data_arrays = classmodel.dataset.dc.to_arrays(
        #     classmodel.graph,
        #     float_dtype=classmodel.float_dtype,
        # )
        # if classmodel.work_arrays is not None:
        #     classmodel._rebuild_work_arrays()

    def mangle(self, data=True, structure=True):
        super().mangle(data=data, structure=structure)
        reset_compiled_methods(self)
        self._is_mangled = True

    def unmangle(self, force=False, structure_only=False):
        if self._is_mangled:
            super().unmangle(force=force, structure_only=structure_only)
            if not structure_only:
                self.reflow_data_arrays()
                self._is_mangled = False
            else:
                self._is_mangled = MANGLE_DATA

    @property
    def dataset(self) -> xr.Dataset | None:
        """xarray.Dataset : Data arrays as loaded for model computation."""
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
        """DataTree : A source for data for the model."""
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

    @property
    def class_membership_model(self):
        # TODO optimize this
        for m in self._models.values():
            if m._model_subtype == "class-membership":
                return m

    def total_weight(self):
        """
        Compute the total weight of cases in the loaded data.

        Returns
        -------
        float
        """
        if self.class_membership_model._data_arrays is not None:
            return self.class_membership_model._data_arrays.wt.sum()
        raise MissingDataError("no data_arrays are set")

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

    def d_logloss(self, x=None, start_case=0, stop_case=-1, step_case=1, **kwargs):
        result = self.d_loglike(
            x, start_case=start_case, stop_case=stop_case, step_case=step_case, **kwargs
        )
        return -np.asarray(result) / self.total_weight()

    def simple_fit_bhhh(self, *args, **kwargs):
        raise NotImplementedError()

    def calculate_parameter_covariance(self, pvals=None, *, robust=False):
        if pvals is None:
            pvals = self.pvals
        locks = np.asarray(self.pholdfast.astype(bool))
        if self.compute_engine == "jax":
            se, hess, ihess = self.jax_param_cov(pvals)
        else:
            raise NotImplementedError(f"compute_engine={self.compute_engine}")
            # hess = -self.d2_loglike(pvals)
            # if self.parameters["holdfast"].sum():
            #     free = self.pholdfast == 0
            #     hess_ = hess[free][:, free]
            #     ihess_ = np.linalg.inv(hess_)
            #     ihess = _arr_inflate(ihess_, locks)
            # else:
            #     ihess = np.linalg.inv(hess)
            # se = np.sqrt(ihess.diagonal())
            # self.pstderr = se
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
        except AttributeError:
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

    def robust_covariance(self):
        raise NotImplementedError()


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
        for _n, k in enumerate(self.class_membership_model.dataset.dc.altids()):
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
        classmodel = self.class_membership_model
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
        seed = self.seed or 42
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
        for _i in range(depth):
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
                seed = self.seed or 0
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
                seed = self.seed or 0
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

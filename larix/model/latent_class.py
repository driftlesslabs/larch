from .param_core import ParameterBucket
from ..compiled import compiledmethod, jitmethod, reset_compiled_methods
from ..optimize import OptimizeMixin
from ..folding import fold_dataset
from .jaxmodel import PanelMixin
import jax.numpy as jnp
import jax
from ..dataset import Dataset, DataTree
import numpy as np
import pandas as pd

class LatentClass(ParameterBucket, OptimizeMixin, PanelMixin):
    compute_engine = 'jax'

    def __init__(self, classmodel, choicemodels, datatree=None, float_dtype=np.float32, **kwargs):
        PanelMixin.__init__(self, **kwargs)
        self._is_mangled = True
        self._dataset = None
        super().__init__(choicemodels, classmodel=classmodel)
        self.datatree = datatree
        assert sorted(classmodel.dataset.dc.altids()) == sorted(choicemodels.keys())
        for choicemodel in choicemodels.values():
            self.dataset = choicemodel.dataset
        if float_dtype is not None:
            for v in self._models.values():
                v.float_dtype = float_dtype

    @property
    def pf(self):
        return self.parameters.to_dataframe()

    @jitmethod
    def jax_probability(self, params):
        classmodel = self['classmodel']
        pr_parts = []
        for n, k in enumerate(self['classmodel'].dataset.dc.altids()):
            pr_parts.append(
                self._models[k].jax_probability(params)
                * jnp.expand_dims(classmodel.jax_probability(params)[..., n], -1)
            )
        return sum(pr_parts)

    @jitmethod
    def jax_loglike_casewise(self, params):
        n_alts = self.dataset.dc.n_alts
        ch = jnp.asarray(self.dataset['ch'])
        if ch.ndim == 2:
            pr = self.jax_probability(params)
            masked_pr = jnp.where(
                ch[..., :n_alts]>0,
                pr[..., :n_alts],
                1.0
            )
            log_pr = jnp.log(masked_pr)
            return (log_pr[...,:n_alts] * ch[...,:n_alts]).sum()
        elif ch.ndim >= 3:
            classmodel = self['classmodel']
            likely_parts = []
            for n, k in enumerate(self['classmodel'].dataset.dc.altids()):
                k_pr = self._models[k].jax_probability(params)
                masked_k_pr = jnp.where(
                    ch[..., :n_alts] > 0,
                    k_pr[..., :n_alts],
                    1.0
                )
                k_likely = jnp.power(masked_k_pr, ch[..., :n_alts]).prod([-2,-1])
                likely_parts.append(
                    k_likely
                    * classmodel.jax_probability(params)[..., 0, n]
                )
            return jnp.log(sum(likely_parts))

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
            raise NotImplementedError(f'latent class with engine={self.compute_engine}')
        if start_case is not None:
            raise NotImplementedError('start_case with engine=jax')
        if stop_case is not None:
            raise NotImplementedError('stop_case with engine=jax')
        if step_case is not None:
            raise NotImplementedError('step_case with engine=jax')
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
            start_case=start_case, stop_case=stop_case, step_case=step_case,
            leave_out=leave_out, keep_only=keep_only, subsample=subsample
        )
        return -result

    @jitmethod
    def jax_d_loglike(self, params):
        return jax.grad(self.jax_loglike)(params)

    def d_loglike(
            self,
            x=None,
            *,
            start_case=None, stop_case=None, step_case=None,
            return_series=False,
            **kwargs,
    ):
        if self.compute_engine != 'jax':
            raise NotImplementedError(f'latent class with engine={self.compute_engine}')
        if start_case is not None:
            raise NotImplementedError('start_case with engine=jax')
        if stop_case is not None:
            raise NotImplementedError('stop_case with engine=jax')
        if step_case is not None:
            raise NotImplementedError('step_case with engine=jax')
        if x is not None:
            self.pvals = x
        result = self.jax_d_loglike(self.pvals)

        print("converge?=", jnp.max(jnp.absolute(result)))

        if return_series:
            result = pd.Series(result, index=self.pnames)
        return result

    def neg_d_loglike(self, x=None, start_case=0, stop_case=-1, step_case=1, **kwargs):
        result = self.d_loglike(
            x,
            start_case=start_case, stop_case=stop_case, step_case=step_case,
            **kwargs
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

        request = self._models['classmodel'].required_data()
        request.pop('avail_any', None)
        for kname, kmodel in self._models.items():
            if kname == 'classmodel':
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
            request['group_co'] = self.groupid

        from .data_arrays import prepare_data
        dataset, self.dataflows = prepare_data(
            datasource=datatree,
            request=request,
            float_dtype=np.float32,
            cache_dir=datatree.cache_dir,
            flows=getattr(self, 'dataflows', None),
        )
        if isinstance(self.groupid, str):
            dataset = fold_dataset(dataset, 'group')
        elif self.groupid is not None:
            dataset = fold_dataset(dataset, self.groupid)
        self.dataset = dataset

        for kname, kmodel in self._models.items():
            if kname == 'classmodel':
                continue
            kmodel.dataset = self.dataset
        self._models['classmodel'].dataset = self.dataset.drop_dims(
            self.dataset.dc.ALTID
        ).dc.set_altids(self._models['classmodel'].dataset.dc.altids())

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
